import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from src import config
from src.dataloader import TestDataLoader
from src.opt import opt
from src.tester import ModelInference
from src.utils import get_pretrain


# def test(model_path, img_path, batch_size, num_classes, keyword):
#     drown_ls = torch.FloatTensor([]).to("cuda:0")
#     stand_ls = torch.FloatTensor([]).to("cuda:0")
#     labels_ls_drown = torch.LongTensor([]).to("cuda:0")
#     labels_ls_stand = torch.LongTensor([]).to("cuda:0")
#     data_loader = DataLoader(img_path, batch_size)
#     pre_name = get_pretrain(model_path)
#     model = ModelInference(num_classes, pre_name, model_path, cfg='default')
#
#     pbar = tqdm(enumerate(data_loader.dataloaders_dict[keyword]), total=len(data_loader.dataloaders_dict[keyword]))
#     for i, (names, inputs, labels) in pbar:
#         inputs = inputs.to("cuda:0")
#         labels = labels.to("cuda:0")
#         outputs = model.CNN_model(inputs)
#         _, preds = torch.max(outputs, 1)
#         drown = torch.index_select(torch.sigmoid(outputs), 1, torch.tensor([1]).to("cuda:0"))
#         drown = drown.view(1, -1).squeeze()
#         stand = torch.index_select(torch.sigmoid(outputs), 1, torch.tensor([0]).to("cuda:0"))
#         stand = stand.view(1, -1).squeeze()
#         drown_ls = torch.cat((drown_ls, drown), 0)
#         stand_ls = torch.cat((stand_ls, stand), 0)
#         labels_ls_drown = torch.cat((labels_ls_drown, labels), 0)
#         labels_ls_stand = torch.cat((labels_ls_stand, torch.add(1, -labels).long()), 0)
#     precision, recall, thresholds = precision_recall_curve(labels_ls_drown.cpu().detach().numpy(),
#                                                            np.array(drown_ls.squeeze().cpu().detach().numpy()))
#     '''#for stand class
#     labels_ls_stand = labels_ls_stand.ge(-0.1)
#     precision, recall, thresholds = precision_recall_curve(labels_ls_stand.cpu().detach().numpy(),
#                                                    np.array(stand_ls.squeeze().cpu().detach().numpy()))
#                                                    '''
#     plt.figure(1)
#     plt.plot(precision, recall)
#     plt.show()


def eval(model_path, img_path, num_classes, keyword, cfg=None):

    label_tensors, preds_tensors = [], []
    for _ in range(num_classes):
        label_tensors.append(torch.LongTensor([]).to("cuda:0"))
        preds_tensors.append(torch.FloatTensor([]).to("cuda:0"))

    data_loader = TestDataLoader(img_path, opt.batch, keyword)
    pbar = tqdm(enumerate(data_loader.dataloaders_dict[keyword]), total=len(data_loader.dataloaders_dict[keyword]))
    pre_name = get_pretrain(model_path)
    Inference = ModelInference(num_classes, pre_name, model_path, cfg=cfg)

    for i, (names, inputs, labels) in pbar:
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
        # model = ModelInference(num_classes, pre_name, model_path, cfg='default')
        outputs = Inference.CNN_model(inputs)
        _, preds = torch.max(outputs, 1)

        for idx in range(num_classes):
            pred = torch.index_select(torch.sigmoid(outputs), 1, torch.tensor([abs(1-idx)]).to("cuda:0"))
            pred = pred.view(1, -1).squeeze()
            preds_tensors[idx] = torch.cat((preds_tensors[idx], pred), 0)
            if idx == 0:
                label_tensors[idx] = torch.cat((label_tensors[idx], labels), 0)
            else:
                label_tensors[idx] = torch.cat((label_tensors[idx], torch.add(1, -labels).long()), 0)

    precision, recall, thresholds = precision_recall_curve(label_tensors[0].cpu().detach().numpy(),
                                                           np.array(preds_tensors[0].squeeze().cpu().detach().numpy()))
    '''#for stand class
    labels_ls_stand = labels_ls_stand.ge(-0.1)
    precision, recall, thresholds = precision_recall_curve(labels_ls_stand.cpu().detach().numpy(),
                                                   np.array(stand_ls.squeeze().cpu().detach().numpy()))
                                                   '''
    plt.figure(1)
    plt.plot(precision, recall)
    plt.show()


if __name__ == '__main__':
    model_dict = {"weight/pruning_test/origin/origin_resnet18_2cls_best.pth": None,
                  "weight/pruning_test/sparse/sparse_resnet18_13_decay1.pth": None,
                  "weight/pruning_test/pruned/new_model_resnet18.pth": "weight/pruning_test/cfg2.txt",
                  "weight/pruning_test/finetune/finetune_resnet18_2cls_best.pth": "weight/pruning_test/cfg2.txt"}
    # model_path = config.eval_model_path
    img_path = config.eval_img_folder
    # model_config = config.eval_config
    keyword = config.eval_keyword
    opt.dataset = "CatDog"
    import os
    num_classes = len(os.listdir(os.path.join(img_path, "train")))

    for model_path, model_config in model_dict.items():
        with torch.no_grad():
            opt.loadModel = model_path
            eval(model_path, img_path, num_classes, keyword, model_config)
