import torch
import numpy as np
import torch.nn as nn
import cv2


def parse_module_defs_shortcut(model):
    ignore_id, normal_id, shortcut_id, downsample_id, all_bn_id = [], [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "bn2" in layer[0]:
                shortcut_id.append(i)
            elif "downsample" in layer[0] or i < 5:
                downsample_id.append(i)
            # elif i < 5:
            #     ignore_id.append(i)
            else:
                normal_id.append(i)
    return normal_id, shortcut_id, downsample_id, all_bn_id


def parse_module_defs(model):
    all_conv = []
    bn_id = []
    ignore_idx = set()
    # add bn module to prune_id and every first conv layer of block like layer *.0.conv1
    # for i, layer in enumerate(list(model.named_modules())):
    #     if isinstance(layer[1], nn.BatchNorm2d):
    #         bn_id.append(i)
    # prune_id = bn_id
    for i, layer in enumerate(list(model.named_modules())):
        # choose the first conv in every basicblock
        if '.bn1' in layer[0]:
            bn_id.append(i)
        if 'conv' in layer[0]:
            all_conv.append(i)
    prune_id = [idx for idx in bn_id if idx not in ignore_idx]
    return prune_id, ignore_idx, all_conv


def gather_bn_weights(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


class BNOptimizer():
    @staticmethod
    def updateBN(model, s):
        # if sr_flag:
        # bn_weights = gather_bn_weights(list(model.named_modules()), prune_idx)
        # bn_numpy = bn_weights.numpy()
        # if np.mean(bn_numpy) < 0.01:
        #     s = s * 0.01

        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.weight.grad.data.add_(s * torch.sign(mod.weight.data))


def obtain_bn_mask(bn_module, thre):
    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def sort_bn(model, prune_idx):
    # size_list = [m[idx][1].weight.data.shape[0] for idx, m in enumerate(list(model.named_modules())) if idx in prune_idx]
    size_list = [m[1].weight.data.shape[0] for idx, m in enumerate(model.named_modules()) if idx in prune_idx]
    # bn_layer = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_prune_layers = [m[1] for idx, m in enumerate(list(model.named_modules())) if idx in prune_idx]
    bn_weights = torch.zeros(sum(size_list))

    index = 0
    for module, size in zip(bn_prune_layers, size_list):
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index += size
    sorted_bn = torch.sort(bn_weights)[0]

    return sorted_bn


def obtain_bn_threshold(sorted_bn, percentage):
    thre_index = int(len(sorted_bn) * percentage)
    thre = sorted_bn[thre_index]
    return thre


def obtain_filters_mask(model, prune_idx, thre):
    pruned = 0
    bn_count = 0
    total = 0
    num_filters = []
    pruned_filters = []
    filters_mask = []
    pruned_maskers = []

    for idx, module in enumerate(list(model.named_modules())):
        if isinstance(module[1], nn.BatchNorm2d):
            # print(idx)
            if idx in prune_idx:
                mask = obtain_bn_mask(module[1], thre).cpu().numpy()
                remain = int(mask.sum())

                if remain == 0:  # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module[1].weight.data.abs().max()
                    mask = obtain_bn_mask(module[1], max_value).cpu().numpy()
                    remain = int(mask.sum())
                    # pruned = pruned + mask.shape[0] - remain

                pruned = pruned + mask.shape[0] - remain
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')

                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())
                total += mask.shape[0]
                num_filters.append(remain)
                filters_mask.append(mask.copy())
            else:

                mask = np.ones(module[1].weight.data.shape)
                remain = mask.shape[0]

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return pruned_filters, pruned_maskers


def init_weights_from_loose_model_shortcut(compact_model, model, CBLidx2mask, downsample_idx):
    layer_nums = [k for k in CBLidx2mask.keys()]
    for idx, layer_num in enumerate(layer_nums):

        out_channel_idx = np.argwhere(CBLidx2mask[layer_num])[:, 0].tolist()
        if idx == 0:
            in_channel_idx = [0,1,2]
        elif layer_num+1 in downsample_idx:
            last_conv_index = layer_nums[idx-3]
            in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()
        else:
            last_conv_index = layer_nums[idx-1]
            in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()

        compact_bn, loose_bn = list(compact_model.named_modules())[layer_num + 1][1], list(model.named_modules())[layer_num + 1][1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

        compact_conv, loose_conv = list(compact_model.named_modules())[layer_num][1], list(model.named_modules())[layer_num][1]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    fc_in_channel_idx = np.argwhere(CBLidx2mask[layer_num])[:, 0].tolist()
    compact_model.fc.weight.data = model.fc.weight.data[:, fc_in_channel_idx].clone()
        # if idx in Conv_id:
        #     # out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
        #     # last conv index
        #     last_conv_index = all_conv_layer[all_conv_layer.index(idx) - 1]
        #     if last_conv_index in Conv_id:
        #         in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()
        #     else:
        #         in_channel_idx = list(range(
        #             list(prune_model.named_modules())[all_conv_layer[all_conv_layer.index(idx) - 1]][
        #                 1].out_channels))
        # else:
        #     # we should make conv2's input equal to the output channel of conv1.
        #     # out_channel_idx = list(range(list(prune_model.named_modules())[idx][1].out_channels))
        #     index = all_conv_layer[all_conv_layer.index(idx) - 1]
        #     in_channel_idx = np.argwhere(CBLidx2mask[index])[:, 0].tolist()

        # compact_bn, loose_bn = list(prune_model.named_modules())[idx + 1][1], list(model.named_modules())[idx + 1][1]
        # compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        # compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        # compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        # compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()
        #
        # compact_conv, loose_conv = list(prune_model.named_modules())[idx][1], list(model.named_modules())[idx][1]
        # tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        # compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()


def init_weights_from_loose_model(all_conv_layer, Conv_id, prune_model, model, CBLidx2mask):
    for i, idx in enumerate(all_conv_layer):
        if i > 0:
            if idx in Conv_id:
                out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
                # last conv index
                last_conv_index = all_conv_layer[all_conv_layer.index(idx) - 1]
                if last_conv_index in Conv_id:
                    in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()
                else:
                    in_channel_idx = list(range(
                        list(prune_model.named_modules())[all_conv_layer[all_conv_layer.index(idx) - 1]][
                            1].out_channels))
            else:
                # we should make conv2's input equal to the output channel of conv1.
                out_channel_idx = list(range(list(prune_model.named_modules())[idx][1].out_channels))
                index = all_conv_layer[all_conv_layer.index(idx) - 1]
                in_channel_idx = np.argwhere(CBLidx2mask[index])[:, 0].tolist()

            compact_bn, loose_bn = list(prune_model.named_modules())[idx + 1][1], list(model.named_modules())[idx + 1][1]
            compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
            compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
            compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
            compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

            compact_conv, loose_conv = list(prune_model.named_modules())[idx][1], list(model.named_modules())[idx][1]
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()


def merge_mask(CBLidx2mask, CBLidx2filter):
    # mask_begin = [23, 39, 55]
    mask_groups = [[1, 10, 16], [23, 26, 32], [39, 42, 48], [55, 58, 64]]
    # for idx, mod in enumerate(model.modules()):
    for layer1, layer2, layer3 in mask_groups:
        Merge_masks = []
        Merge_masks.append(torch.Tensor(CBLidx2mask[layer1]).unsqueeze(0))
        Merge_masks.append(torch.Tensor(CBLidx2mask[layer2]).unsqueeze(0))
        Merge_masks.append(torch.Tensor(CBLidx2mask[layer3]).unsqueeze(0))
        Merge_masks = torch.cat(Merge_masks, 0)
        merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()

        filter_num = int(torch.sum(merge_mask).item())
        merge_mask = np.array(merge_mask)
        CBLidx2mask[layer1] = merge_mask
        CBLidx2mask[layer2] = merge_mask
        CBLidx2mask[layer3] = merge_mask

        CBLidx2filter[layer1] = filter_num
        CBLidx2filter[layer2] = filter_num
        CBLidx2filter[layer3] = filter_num
    return CBLidx2mask, CBLidx2filter

    # for i in range(len(model.module_defs) - 1, -1, -1):
    #     mtype = model.module_defs[i]['type']
    #     if mtype == 'shortcut':
    #         if model.module_defs[i]['is_access']:
    #             continue
    #
    #         Merge_masks = []
    #         layer_i = i
    #         while mtype == 'shortcut':
    #             model.module_defs[layer_i]['is_access'] = True
    #
    #             if model.module_defs[layer_i - 1]['type'] == 'convolutional':
    #                 bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
    #                 if bn:
    #                     Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))
    #
    #             layer_i = int(model.module_defs[layer_i]['from']) + layer_i
    #             mtype = model.module_defs[layer_i]['type']
    #
    #             if mtype == 'convolutional':
    #                 bn = int(model.module_defs[layer_i]['batch_normalize'])
    #                 if bn:
    #                     Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))
    #
    #         if len(Merge_masks) > 1:
    #             Merge_masks = torch.cat(Merge_masks, 0)
    #             merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
    #         else:
    #             merge_mask = Merge_masks[0].float()
    #
    #         layer_i = i
    #         mtype = 'shortcut'
    #         while mtype == 'shortcut':
    #
    #             if model.module_defs[layer_i - 1]['type'] == 'convolutional':
    #                 bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
    #                 if bn:
    #                     CBLidx2mask[layer_i - 1] = merge_mask
    #                     CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())
    #
    #             layer_i = int(model.module_defs[layer_i]['from']) + layer_i
    #             mtype = model.module_defs[layer_i]['type']
    #
    #             if mtype == 'convolutional':
    #                 bn = int(model.module_defs[layer_i]['batch_normalize'])
    #                 if bn:
    #                     CBLidx2mask[layer_i] = merge_mask
    #                     CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())


def image_normalize(img_name, size=224):
    image_normalize_mean = [0.485, 0.456, 0.406]
    image_normalize_std = [0.229, 0.224, 0.225]
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor


def predict(CNN_model, img):
    img_tensor_list = []
    img_tensor = image_normalize(img)
    img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
    if len(img_tensor_list) > 0:
        input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
        res_array = predict_image(CNN_model, input_tensor)
        return res_array


def predict_image(CNN_model, image_batch_tensor):
    CNN_model.eval()
    image_batch_tensor = image_batch_tensor.cuda()
    outputs = CNN_model(image_batch_tensor)
    outputs_tensor = outputs.data
    m_softmax = nn.Softmax(dim=1)
    outputs_tensor = m_softmax(outputs_tensor).to("cpu")
    print(outputs)
    return np.asarray(outputs_tensor)
