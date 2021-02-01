from src.tester import ModelInference
import os
import cv2
from src import config, utils

classes = ["cat", "dog"]
num_classes = len(classes)
colors = {"cat": (0,255,255), "dog":(255,255,0)}


class Tester:
    def __init__(self, model_path, conf=0.5):
        self.pre_name = utils.get_pretrain(model_path)
        self.model = ModelInference(num_classes, self.pre_name, model_path,cfg=[])

    def test_score(self, img):
        score = self.model.predict(img)
        return score

    def test_idx(self, img):
        score = self.test_score(img)
        idx = score[0].tolist().index(max(score[0].tolist()))
        return idx

    def test_pred(self, img):
        idx = self.test_idx(img)
        pred = classes[idx]
        return pred

    def show_img(self, img, pred):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, pred, (50, 50), font, 2, colors[pred], 3)
        cv2.imshow("Result", img)
        cv2.waitKey(500)

    def to_onnx(self):
        self.model.to_onnx()

    def to_libtorch(self):
        self.model.to_libtorch()


if __name__ == '__main__':
    model_pth = config.test_model_path
    img_path = config.test_img_folder
    # batch_size = 16
    # data_loader = DataLoader(img_path, batch_size)
    # for names, inputs, labels in data_loader.dataloaders_dict['val']:
    #     inputs = inputs.to("cuda:0")
    #     labels = labels.to("cuda:0")
    MI = Tester(model_pth)
    # max_idx = MI.test_idx(cv2.imread(img_path))
    # print(max_idx)
    for img_name in os.listdir(img_path):
        im = cv2.imread(os.path.join(img_path, img_name))
        pred = MI.test_pred(im)
        MI.show_img(im, pred)
        print("Prediction of {} is {}".format(img_name, pred))
    MI.to_libtorch()
