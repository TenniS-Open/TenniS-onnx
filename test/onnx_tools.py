import sys
import time
import cv2
import numpy as np
tennis = '/home/kier/git/TensorStack/python'
sys.path.append(tennis)

from tennis.backend.api import *
from tennis_onnx import onnx_tools as tools


def load_json_pre_processor(path):
    with open(path, "r") as f:
        import json
        obj = json.load(f)
        assert "pre_processor" in obj
        return obj["pre_processor"]


class ExampleImageFilter(object):
    def __init__(self):
        device = Device("cpu")
        self.__workbench = Workbench(device=device)
        self.__image_filter = ImageFilter(device=device)
        self.__image_filter.to_float()
        self.__image_filter.to_chw()

    def dispose(self):
        self.__image_filter.dispose()
        self.__workbench.dispose()

    def __call__(self, image):
        self.__workbench.setup_context()

        image = image[0]
        # do image actions

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = image[:, :, numpy.newaxis]

        image = numpy.expand_dims(image, 0)

        input = Tensor(image)
        output = self.__image_filter.run(input)
        output_numpy = output.numpy
        input.dispose()
        output.dispose()
        return output_numpy


def test():
    working_root = "/home/seeta/Documents/models/"
    img_path = "debug.jpg"
    input_module = working_root + "model_arcface_2020-3-4.tsm"

    tools.test_tsm_with_onnx(input_module,
                             None,
                             None,
                             img_path,
                             [
                                 {"op": "resize", "size": [112, 112]},
                                 {"op": "to_float"},
                                 {"op": "sub_mean", "mean": [0.0, 0.0, 0.0]},
                                 {"op": "div_std", "std": [255.0, 255.0, 255.0]},
                                 {"op": "to_chw"},
                             ])


if __name__ == '__main__':
    test()
