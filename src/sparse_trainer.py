# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import torch
import time
import src.config as config
from src.config import device, datasets, input_size, computer, patience_decay
import os
import cv2
from src import utils
from src.opt import opt
from apex import amp
import copy
from src.utils import warm_up_lr, lr_decay,lr_decay2, EarlyStopping, write_decay_title, write_decay_info, log_of_each_class, \
    write_csv_title, csv_cls_num
from prune.utils import *
import logging
logging.basicConfig(level=logging.DEBUG,
                    filename='new.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
import csv

record_num = 1
label_dict = datasets[opt.dataset]
opt.classes = ",".join(label_dict)
class_nums = len(label_dict)
warm_up_epoch = max(config.warm_up.keys())


def train_model(model, dataloaders, criterion, optimizer, cmd, writer, is_inception=False, model_save_path="./"):
    print("-------------------sparse training-----------------------")
    num_epochs = opt.epoch
    log_dir = os.path.join(model_save_path, opt.expID)
    os.makedirs(log_dir, exist_ok=True)
    log_save_path = os.path.join(log_dir, "log.txt")
    since = time.time()
    best_weight = copy.deepcopy(model)
    val_acc_history, train_acc_history, val_loss_history, train_loss_history = [], [], [], []
    train_acc, val_acc, train_loss, val_loss, best_epoch, epoch_acc, epoch = 0, 0, float("inf"), float("inf"), 0, 0, 0
    epoch_ls = []
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True)

    prune_idx,ignore_id,all_conv = parse_module_defs(model)
    print(prune_idx)

    decay, decay_epoch = 0, []
    stop = False
    log_writer = open(log_save_path, "w")
    log_writer.write(cmd)
    log_writer.write("\n")
    lr = opt.LR

    train_log_name = log_save_path.replace("log.txt", "train_log.csv")
    train_log = open(train_log_name, "w", newline="")
    csv_writer = csv.writer(train_log)
    csv_writer.writerow(write_csv_title())

    os.makedirs("result", exist_ok=True)
    result = os.path.join("result", "{}_result_{}.csv".format(opt.expFolder, computer))
    exist = os.path.exists(result)

    print("----------------------------------------------------------------------------------------------------")
    print(opt)
    print("Training backbone is: {}".format(opt.backbone))
    print("Warm up end at {}".format(warm_up_epoch))
    for k, v in config.bad_epochs.items():
        if v > 1:
            raise ValueError("Wrong stopping accuracy!")
    print("----------------------------------------------------------------------------------------------------")

    utils.draw_graph(epoch_ls, train_loss_history, val_loss_history, train_acc_history, val_acc_history, log_dir)
    flops = utils.print_model_param_flops(model)
    print("FLOPs of current model is {}".format(flops))
    params = utils.print_model_param_nums(model)
    print("Parameters of current model is {}".format(params))
    inf_time = utils.get_inference_time(model, height=input_size, width=input_size)
    print("Inference time is {}".format(inf_time))
    print("----------------------------------------------------------------------------------------------------")

    for epoch in range(num_epochs):
        log_tmp = [opt.expID, epoch]

        if epoch < warm_up_epoch:
            optimizer, lr = warm_up_lr(optimizer, epoch)
        elif epoch == warm_up_epoch:
            lr = opt.LR
        elif epoch> num_epochs * 0.7 and epoch< num_epochs *0.9:
            optimizer, lr = lr_decay(optimizer, lr)
        elif epoch> num_epochs *0.9:
            optimizer, lr = lr_decay2(optimizer, lr)


        log_tmp.append(lr)
        log_tmp.append("")

        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        log_writer.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        log_writer.write('-' * 10 + "\n")

        writer.add_scalar("lr", lr, epoch)
        print("Current lr is {}".format(lr))

        for name, param in model.named_parameters():
            writer.add_histogram(
                name, param.clone().data.to("cpu").numpy(), epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            cls_correct = [0] * class_nums
            cls_sum = [0] * class_nums
            cls_acc = [0] * class_nums

            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch_num = 0
            batch_start_time = time.time()
            for names, inputs, labels in dataloaders[phase]:
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
                # print(len(inputs))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizer, lr = utils.adjust_lr(optimizer, epoch, opt.epoch)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    right = preds == labels
                    for r, l in zip(right, labels):
                        cls_sum[l] += 1
                        if r:
                            cls_correct[l] += 1

                    if phase == 'train':
                        if opt.mix_precision:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        # sr_flag = True
                        s = opt.sparse_s
                        # BNOptimizer.updateBN(sr_flag, model, s, prune_idx)
                        BNOptimizer.updateBN(model, s, prune_idx)


                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if batch_num % 100 == 0:
                    print("batch num:", batch_num, "cost time:", time.time() - batch_start_time)
                    batch_start_time = time.time()
                batch_num += 1

            for idx, (s, c) in enumerate(zip(cls_sum, cls_correct)):
                cls_acc[idx] = c / s

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            bn_weights = gather_bn_weights(list(model.named_modules()), prune_idx)
            bn_numpy = bn_weights.numpy()
            print(np.mean(bn_numpy))


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_writer.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                log_tmp.insert(5, epoch_acc.tolist())
                log_tmp.insert(6, epoch_loss)

                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_loss = epoch_loss if epoch_loss < val_loss else val_loss

                writer.add_scalar("scalar/val_acc", epoch_acc, epoch)
                writer.add_scalar("Scalar/val_loss", epoch_loss, epoch)
                imgnames, pds = names[:3], [label_dict[i] for i in preds[:record_num].tolist()]
                for idx, (img_path, pd) in enumerate(zip(imgnames, pds)):
                    img = cv2.imread(img_path)
                    img = cv2.putText(img, pd, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    #cv2.imwrite("tmp/{}_{}.jpg".format(epoch, idx), img)
                    tb_img = utils.image2tensorboard(img)
                    # images = torch.cat((images, torch.unsqueeze(tb_img, 0)), 0)
                    writer.add_image("pred_image_for_epoch{}".format(epoch), tb_img, epoch)

                if epoch % opt.save_interval == 0 and epoch != 0:
                    torch.save(model.state_dict(),
                               os.path.join(model_save_path, "{}_{}_{}cls_{}.pth".format(
                                   opt.expID, opt.backbone, class_nums, epoch)))

                # writer.add_image("pred_image_for_epoch{}".format(epoch), images[1:, :, :, :])
                if epoch_acc > val_acc:
                    torch.save(model.state_dict(),
                               os.path.join(model_save_path, "{}_{}_{}cls_best.pth".format(
                                   opt.expID, opt.backbone, class_nums)))
                    val_acc = epoch_acc
                    best_epoch = epoch
                    best_weight = copy.deepcopy(model)

            else:
                log_tmp.append(epoch_acc.tolist())
                log_tmp.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_acc = epoch_acc if epoch_acc > train_acc else train_acc
                train_loss = epoch_loss if epoch_loss < train_loss else train_loss
                writer.add_scalar("scalar/train_acc", epoch_acc, epoch)
                writer.add_scalar("Scalar/train_loss", epoch_loss, epoch)
            log_tmp += log_of_each_class(cls_acc)

        epoch_ls.append(epoch)
        epoch_time_cost = time.time() - epoch_start_time
        print("epoch complete in {:.0f}m {:.0f}s".format(epoch_time_cost // 60, epoch_time_cost % 60))
        log_writer.write(
            "epoch complete in {:.0f}m {:.0f}s\n".format(epoch_time_cost // 60, epoch_time_cost % 60))
        torch.save(opt, '{}/option.pth'.format(model_save_path))
        csv_writer.writerow(log_tmp)

    csv_writer.writerow([])
    csv_writer.writerow(csv_cls_num(dataloaders))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(val_acc))

    log_writer.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log_writer.write('Best val Acc: {:.4f}\n'.format(val_acc))
    log_writer.close()

    with open(result, "a+") as f:
        if not exist:
            title_str = "id,backbone,params,flops,time,batch_size,optimizer,freeze_bn,freeze,sparse,sparse_decay," \
                        "epoch_num,LR,weightDecay,loadModel,location, ,folder_name,train_acc,train_loss,val_acc," \
                        "val_loss,training_time, best_epoch,total_epoch\n"
            title_str = write_decay_title(len(decay_epoch), title_str)
            f.write(title_str)
        info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{},{},{},{},{},{}\n".format(
            opt.expID, opt.backbone, params, flops, inf_time, opt.batch, opt.optMethod,  opt.freeze_bn, opt.freeze,
            opt.sparse_s, opt.sparse_decay, opt.epoch, opt.LR, opt.weightDecay, opt.loadModel, computer,
            os.path.join(opt.expFolder, opt.expID), train_acc, train_loss, val_acc, val_loss,time_elapsed,
            best_epoch, epoch)
        info_str = write_decay_info(decay_epoch, info_str)
        f.write(info_str)

