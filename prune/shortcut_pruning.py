from prune import prune_config as config
from prune.utils import *
from src.tester import ModelInference
from src.opt import opt


def shortcut_prune(model_path, save_cfg_path, save_model_path):
    model_name = config.model_name
    classes = config.classes
    num_classes = len(classes)
    thresh = config.thresh
    opt.loadModel = model_path
    model = ModelInference(num_classes, model_name, model_path, cfg=None).CNN_model
    normal_id, shortcut_id, downsample_id, ignore_id, all_bn_id = parse_module_defs_shortcut(model)
    sorted_bn = sort_bn(model, all_bn_id)
    threshold = obtain_bn_threshold(sorted_bn, thresh / 100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, all_bn_id, threshold)

    CBLidx2mask = {idx-1: mask.astype('float32') for idx, mask in zip(all_bn_id, pruned_maskers)}
    CBLidx2filter = {idx-1: num_filter for idx, num_filter in zip(all_bn_id, pruned_filters)}
    prune_model = ModelInference(num_classes, model_name, None, cfg=None).CNN_model
    CBLidx2mask, CBLidx2filter = merge_mask(CBLidx2mask, CBLidx2filter)

    # init_weights_from_loose_model_shortcut(prune_model, model, CBLidx2mask)

    cfg = [list(layer[1].weight.shape[1::-1]) for layer in list(prune_model.named_modules()) if 'conv' in layer[0]][
          1:]
    # cfg1 = [layer[1].weight.shape[0] for layer in list(prune_model.named_modules()) if 'conv1' in layer[0]][1:]
    cfg1 = ",".join([str(layer[1].weight.shape[0]) for layer in list(prune_model.named_modules()) if 'conv1' in layer[0]][1:])
    with open(save_cfg_path, 'w') as cfg_file:
        print(cfg1, file=cfg_file)
    torch.save(prune_model.state_dict(), save_model_path)


def test_prune_model(test_model_path, cfg, img_path):
    from src.opt import opt
    opt.loadModel = test_model_path
    model = ModelInference(config.num_classes, config.model_name, test_model_path, cfg=cfg).CNN_model
    img = cv2.imread(img_path)
    res = predict(model, img)
    print(res)


if __name__ == '__main__':
    print("begin to prune")
    shortcut_prune(model_path=config.model_path, save_cfg_path=config.pruned_cfg_file,
                 save_model_path=config.pruned_model_file)
    # print("begin to test")
    # test_prune_model(config.pruned_model_file, config.pruned_cfg_file, config.img_path)
