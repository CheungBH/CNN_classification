# -*- coding:utf-8 -*-
from __future__ import print_function
from src.model import CNNModel, LeNet
from src.opt import opt
if opt.sparse_s:
    from src.sparse_trainer import train_model
else:
    from src.trainer import train_model
from src.dataloader import DataLoader
import src.config as config
import torch.nn as nn
import torch.optim as optim
import os
import torch
import sys
from src import utils
from tensorboardX import SummaryWriter
import traceback

try:
    from apex import amp
    opt.mix_precision = True
except ImportError:
    opt.mix_precision = False


device = config.device
feature_extract = opt.freeze
num_epochs = opt.epoch
data_name = opt.dataset

class_nums = len(config.datasets[data_name])
data_dir = os.path.join("data", data_name)
backbone = opt.backbone
batch_size = opt.batch

modelID = opt.expID
model_save_path = os.path.join("weight/{}/{}".format(opt.expFolder, modelID))
os.makedirs(model_save_path, exist_ok=True)

if __name__ == "__main__":
    cmd_ls = sys.argv[1:]
    cmd = utils.generate_cmd(cmd_ls)
    if "--freeze_bn False" in cmd:
        opt.freeze_bn = False

    if backbone == "LeNet":
        model = LeNet(class_nums).to(device)
    else:
        model = CNNModel(class_nums, backbone, feature_extract, cfg=opt.resnet18_cfg).model.to(device)

    params_to_update = model.parameters()

    if opt.freeze > 0 or opt.freeze_bn:
        try:
            feature_layer_num = config.freeze_pretrain[opt.backbone][0]
            classifier_layer_name = config.freeze_pretrain[opt.backbone][1]
            feature_num = int(opt.freeze * feature_layer_num)

            for idx, (n, p) in enumerate(model.named_parameters()):
                if len(p.shape) == 1 and opt.freeze_bn:
                    p.requires_grad = False
                elif classifier_layer_name not in n and idx < feature_num:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        except:
            raise ValueError("This model is not supported for freezing now")

    params_to_update, layers = [], 0
    for name, param in model.named_parameters():
        layers += 1
        if param.requires_grad:
            params_to_update.append(param)
            # print("\t", name)

    print("Training {} layers out of {}".format(len(params_to_update), layers))

    if opt.optMethod == "adam":
        optimizer_ft = optim.Adam(params_to_update, lr=opt.LR, weight_decay=opt.weightDecay)
    elif opt.optMethod == 'rmsprop':
        optimizer_ft = optim.RMSprop(params_to_update, lr=opt.LR, momentum=opt.momentum, weight_decay=opt.weightDecay)
    elif opt.optMethod == 'sgd':
        optimizer_ft = optim.SGD(params_to_update, lr=opt.LR, momentum=opt.momentum, weight_decay=opt.weightDecay)
    else:
        raise ValueError("This optimizer is not supported now")

    writer = SummaryWriter('weight/{}/{}'.format(opt.expFolder, opt.expID), comment=cmd)

    is_inception = backbone == "inception"
    try:
        if is_inception:
            writer.add_graph(model, torch.rand(1, 3, 299, 299).to(device))
        else:
            writer.add_graph(model, torch.rand(1, 3, 224, 224).to(device))
    except:
        pass

    if opt.mix_precision:
        m, optimizer = amp.initialize(model, optimizer_ft, opt_level="O1")

    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(data_dir, batch_size)
    print(data_dir)
    try:
        train_model(model, data_loader.dataloaders_dict, criterion, optimizer_ft, cmd, writer, is_inception=is_inception,
                    model_save_path=model_save_path)
    except:
        if os.path.exists('error.txt'):
            os.remove('error.txt')
        with open('error.txt', 'a+') as f:
            f.write(opt.expID)
            f.write('\n')
            f.write('----------------------------------------------\n')
            traceback.print_exc(file=f)
    print("Model {} training finished".format(modelID))
