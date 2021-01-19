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


def test(model_path, img_path, batch_size, num_classes, keyword):

    label_tensors, preds_tensors = [], []
    for _ in range(num_classes):
        label_tensors.append(torch.LongTensor([]).to("cuda:0"))
        preds_tensors.append(torch.FloatTensor([]).to("cuda:0"))

    data_loader = TestDataLoader(img_path, batch_size, keyword)
    pbar = tqdm(enumerate(data_loader.dataloaders_dict[keyword]), total=len(data_loader.dataloaders_dict[keyword]))
    pre_name = get_pretrain(model_path)
    Inference = ModelInference(num_classes, pre_name, model_path, cfg='default')

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
    model_path = config.test_model_path
    img_path = config.test_img
    keyword = "val"
    batch_size = opt.batch
    opt.dataset = "CatDog"
    opt.loadModel = model_path
    import os
    classes = os.listdir(os.path.join(img_path, "train"))
    num_classes = len(classes)

    with torch.no_grad():
        test(model_path,img_path,batch_size,num_classes, keyword)
