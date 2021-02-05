model_name = 'resnet18'
classes = ['drown', 'stand']
num_classes = len(classes)
thresh = 60
model_path = '../weight/shortcut_prune/sparse/train_resnet18_2cls_best.pth'
pruned_cfg_file = './cfg2.txt'
pruned_model_file = './new_model.pth'

'''----------------------------------------------------------------------------'''

test_model_path = './new_model.pth'
img_path = '../test/CatDog/177.jpg'
