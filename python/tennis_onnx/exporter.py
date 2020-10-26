#!/usr/bin/env python

from typing import Union
import tennis as ts

from typing import Tuple, List, Dict, Optional, Iterable
import numpy
import os

from tennisfence.spliter import MainGraph
from tennisbuilder.export import fridge
from . import onnx_spliter
from . import onnx_fence

from . import onnx_graph
from tennisbuilder.export import dumper
import copy

import sys
if sys.version > "3":
    basestring = str

"""
For add new converter, See .onnx_spliter.get_spliter for graph spliter; .onnx_caffe for graph converter
"""


def split_onnx(input_tsm, output_tsm, subdir=None, input_shape=None, export_main=False, opset_version=9):
    # type: (Union[str, ts.Module], str, str, Union[List[Tuple[int]], Dict[str, Tuple[int]]], bool, int) -> [None, MainGraph]
    """
    Split support node to sub graph.
    :param input_tsm:
    :param output_tsm:
    :param subdir:
    :return:
    Notice: output main tsm module and sub onnx models
    Every output onnx named {output_tsm}.<i>.onnx
    """
    assert isinstance(subdir, (type(None), basestring))
    module = input_tsm
    if isinstance(module, basestring):
        with open(module, "rb") as f:
            module = ts.Module.Load(f)
    assert isinstance(module, ts.Module)
    filepath = os.path.abspath(output_tsm)
    output_root, filename_ext = os.path.split(filepath)
    filename, ext = os.path.splitext(filename_ext)

    output_onnx_root = output_root
    if subdir is not None:
        output_onnx_root = os.path.join(output_root, subdir)

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    if not os.path.isdir(output_onnx_root):
        os.makedirs(output_onnx_root)

    outputs = module.outputs
    inputs = module.inputs
    print("[INFO]: Freezing graph...")
    outputs, inputs = fridge.freeze(outputs, inputs, input_shape)
    print("[INFO]: Split graph...")
    outputs, inputs = onnx_fence.get_fence().convert(outputs, after=inputs)
    main_graph = onnx_spliter.get_spliter().split(outputs, inputs)
    print("[INFO]: Convert graph...")
    sub_graph_count = main_graph.sub_count()
    for i in range(sub_graph_count):
        output_name_body = "{}.{}".format(filename, i)
        print("[INFO]: Exporting... {}.onnx".format(
            os.path.relpath(os.path.join(output_onnx_root, output_name_body), output_root)))
        output_onnx = "{}.onnx".format(output_name_body)
        sub_node = main_graph.sub_node(i)
        sub_graph = main_graph.sub_graph(i)
        onnx_graph.convert(sub_graph.outputs, sub_graph.inputs,
                           os.path.join(output_onnx_root, output_onnx),
                           version=opset_version)

    if export_main:
        print("[INFO]: Exporting... {}".format(filepath))
        main_module = ts.Module()
        main_module.load(main_graph.outputs)
        main_module.sort_inputs(main_graph.inputs)

        with open(filepath, "wb") as f:
            ts.Module.Save(f, main_module)

    return main_graph


def export_image_list(module, output_names, calibrator, main, output_root, cache=None, device="cpu", device_id=0):
    # type: (ts.Module, List[List[str]], dumper.Calibrator, str, str, str, str, int) -> List[str]
    output_root = os.path.join(output_root, main)

    output_root = os.path.abspath(output_root)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    map_unique_output_names = {}
    for name_list in output_names:
        for name in name_list:
            fixed_name = name.replace("/", "=")
            fixed_name = fixed_name.replace("\\", "=")
            map_unique_output_names[name] = fixed_name
    unique_output_names = list(map_unique_output_names.keys())
    map_output_name_index = {}
    for i, name in enumerate(unique_output_names):
        map_output_name_index[name] = i
    list_output_name_path = [os.path.join("npy", "{}".format(i)) for i in range(len(unique_output_names))]
    list_feature_npy = [[], ] * len(unique_output_names)

    # build show data
    if calibrator.number() == 0:
        raise Exception("calibrator.number() must great than 0")

    P = [0, calibrator.number()]

    def process_show():
        sys.stdout.write("\r[{}/{}]   ".format(P[0], P[1]))
        sys.stdout.flush()

    # extract feature
    extractor = dumper.Dumper(module, unique_output_names, calibrator, 1, cache=cache, device=device, device_id=device_id)

    for filepath in list_output_name_path:
        fullpath = os.path.join(output_root, filepath)
        if not os.path.isdir(fullpath):
            os.makedirs(fullpath)

    process_show()

    procceed = 0
    N = 100
    while True:
        features_list = extractor.next()
        if features_list is None:
            break
        for i, name in enumerate(unique_output_names):
            feature_npy = "{}/{:05d}.npy".format(list_output_name_path[i], procceed)
            feature_data = features_list[i]
            feature_data = feature_data.transpose([0, 2, 3, 1])  # save as NHWC format
            numpy.save(os.path.join(output_root, feature_npy), feature_data)
            list_feature_npy[i].append(feature_npy)

        procceed += 1

        P[0] += 1
        process_show()

    process_show()
    print("\n[INFO]: Extract image features done.")

    # write filelist
    dataset_list = []
    for i, name_list in enumerate(output_names):
        sub_graph_dataset_filename = \
            os.path.join(output_root,
                         "{}_[{}].txt".format(i, ",".join([map_unique_output_names[name] for name in name_list])))

        index_list = [map_output_name_index[name] for name in name_list]
        block = '\n'.join([' '.join(npy_files) for npy_files in zip(*[list_feature_npy[i] for i in index_list])])

        with open(sub_graph_dataset_filename, "w") as f:
            f.write(block)
            f.write("\n")

        dataset_list.append(sub_graph_dataset_filename)
    print("[INFO]: Build dataset file list done.")

    return [os.path.join(output_root, path) for path in dataset_list]


class Calibrator(object):
    """
    Return valid set
    """
    def next(self):
        # type: () -> Tuple[numpy.ndarray]
        """
        Get next sample for quantification, tuple means multi inputs
        :return:
        """
        raise NotImplementedError


class NetInferer(object):
    def run(self, inputs, outputs):
        # type: (List[numpy.ndarray], List[str]) -> List[numpy.ndarray]
        """
        :param inputs: length input count
        :param outputs: get output names
        :return:
        """
        raise NotImplementedError


def _check_input_shape_dict_str_int_list(shape):
    # type: (Dict[str, Tuple[int]]) -> bool
    if not isinstance(shape, dict):
        return False
    for k, v in shape.items():
        if not isinstance(k, str):
            return False
        if not _check_input_shape_int_list(v):
            return False
    return True


def _check_input_shape_int_list(shape):
    # type: (Iterable[int]) -> bool
    if not isinstance(shape, (list, tuple)):
        return False
    for i in shape:
        if not isinstance(i, int):
            return False
    return True


def _check_input_shape_list_of_int_list(shape):
    # type: ( List[Iterable[int]]) -> bool
    if not isinstance(shape, (list, tuple)):
        for i in shape:
            if not _check_input_shape_int_list(i):
                return False
    return True


def _check_input_shape(shape):
    # type: (Union[Iterable[int], List[Iterable[int]], Dict[str, Iterable[int]]]) -> Union[List[Iterable[int]], Dict]
    def _error():
        raise Exception("Input shape must be List[int], List[Tuple[int]] or Dict[str, Tuple[int]]")

    if isinstance(shape, dict):
        if not _check_input_shape_dict_str_int_list(shape):
            _error()
        return shape

    if _check_input_shape_int_list(shape):
        return [shape]

    if not _check_input_shape_list_of_int_list(shape):
        _error()

    return shape


class ONNXConfig(object):
    def __init__(self):
        self.__opset_version = 9

    @property
    def opset_version(self):
        return self.__opset_version

    @opset_version.setter
    def opset_version(self, value):
        assert isinstance(value, int) and value >= 9
        self.__opset_version = value

    def tag(self):
        return "OPV{}".format(self.__opset_version)


class ONNXExporter(object):
    def __init__(self, host_device="cpu", host_device_id=0):
        self.__original_module = None   # update by load
        self.__input_shape = None       # update by load

        self.__cache = None # cache temp files
        self.__host_device = host_device
        self.__host_device_id = host_device_id

        self.__max_batch_size = 1   # default max batch size

        self.__config = ONNXConfig()
        pass

    @property
    def max_batch_size(self):
        return self.__max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value):
        value = int(value)
        if not 1 <= value <= 256:
            raise ValueError("max_batch_size must be in [1, 256]")
        self.__max_batch_size = value

    @property
    def config(self):
        # type: () -> ONNXConfig
        return self.__config

    @config.setter
    def config(self, val):
        assert isinstance(val, ONNXConfig)
        self.__config = val

    def load(self, module, input_shape=None):
        # type: (Union[str, ts.Module], Union[List[Iterable[int]], Dict[str, Iterable[int]]]) -> None
        if isinstance(module, basestring):
            print("[INFO]: Loading... {}".format(module))
            with open(module, "rb") as f:
                module = ts.Module.Load(f)
        assert isinstance(module, ts.Module)
        self.__original_module = module
        self.__input_shape = input_shape
        # check input shape must be valid
        if input_shape is not None:
            input_shape = _check_input_shape(input_shape)
            if isinstance(input_shape, (list, tuple)):
                for shape in input_shape:
                    for dim in shape[1:]:
                        if dim <= 0:
                            raise ValueError("Input shape must be definite, got {}".format(input_shape))
            elif isinstance(input_shape, dict):
                for shape in input_shape.values():
                    for dim in shape[1:]:
                        if dim <= 0:
                            raise ValueError("Input shape must be definite, got {}".format(input_shape))

    def export_onnx(self, filename, subdir=None, export_main=True):
        # type: (str, str, bool) -> None
        if self.__original_module is None:
            raise ValueError("call load fist be before export_onnx")

        output_root, output_name, output_ext = self._split_root_name_ext(filename)
        # 1. split caffe
        main_graph = split_onnx(self.__original_module, filename, subdir, self.__input_shape, export_main=False,
                                opset_version=self.config.opset_version)
        if not export_main:
            return

        # 2. get image list
        sub_graph_inputs = set()
        sub_graph_count = main_graph.sub_count()
        for i in range(sub_graph_count):
            for input in main_graph.sub_graph(i).inputs:
                sub_graph_inputs.add(input.name)
        sub_graph_inputs = list(sub_graph_inputs)

        summery_configs = []
        # 3. write nnie cfg file
        for i in range(sub_graph_count):
            node = main_graph.sub_node(i)
            graph = main_graph.sub_graph(i)

            # ref wk filename
            # rknn_instruction_name = os.path.join("rknn", "{}.{}".format(output_name, i))
            # rknn_filename = rknn_instruction_name + ".rknn"
            # print("[INFO]: Waiting... {}".format(rknn_filename))

            onnx_filename = os.path.join(subdir, "{}.{}.onnx".format(output_name, i))

            # update node
            node.op = "onnx"
            node.set("input_count", len(graph.inputs), numpy.int32)     # required
            node.set("output_count", len(graph.outputs), numpy.int32)   # required
            node.set("onnx_file", onnx_filename)    # required

        # 4. write main tsm file
        main_module = ts.Module()
        main_module_outputs, main_module_inputs = \
            onnx_fence.back_fence().convert(main_graph.outputs, after=main_graph.inputs)
        main_module.load(main_module_outputs)
        main_module.sort_inputs(main_module_inputs)

        if not os.path.isdir(output_root):
            os.makedirs(output_root)

        with open(filename, "wb") as f:
            ts.Module.Save(f, main_module)

        print("[INFO]: Writen file: {}".format(filename))

    def _split_root_name_ext(self, filename):
        # type: (str) -> Tuple[str, str, str]
        filepath = os.path.abspath(filename)
        root, name_ext = os.path.split(filepath)
        name, ext = os.path.splitext(name_ext)
        return root, name, ext

    @staticmethod
    def FuseONNX(input_filename, output_filename):
        # type: (str, str) -> None
        """
        Fuse all nnie operators' wk_file to wk_buffer
        :param input_filename:
        :param output_filename:
        :return:
        """
        input_root = os.path.split(os.path.abspath(input_filename))[0]
        # output_root = os.path.split(os.path.abspath(output_filename))[0]
        with open(input_filename, "rb") as f:
            input_module = ts.Module.Load(f)
        input_graph, _ = ts.graph.walk_graph(input_module.outputs)
        for node in input_graph:
            if node.op == "onnx":
                wk_file = str(node.get("onnx_file"))
                abs_wk_file = os.path.join(input_root, wk_file)
                if not os.path.isfile(abs_wk_file):
                    raise FileNotFoundError("File {} not found in {}".format(wk_file, input_filename))
                # merge file in wk_file
                with open(abs_wk_file, "rb") as f:
                    wk_buffer = f.read()
                # buffer to tensor
                dtype_numpy = numpy.dtype(numpy.uint8)
                dtype_numpy = dtype_numpy.newbyteorder('<')
                tensor = numpy.frombuffer(wk_buffer, dtype=dtype_numpy)
                tensor = numpy.reshape(tensor, [-1])
                node.set("onnx_buffer", tensor)

        output_module = input_module
        with open(output_filename, "wb") as f:
            ts.Module.Save(f, output_module)

    def suggest_name(self, output_root, input_filename):
        path, name_ext = os.path.split(input_filename)
        name, ext = os.path.splitext(name_ext)
        return os.path.join(output_root, "{}.{}{}".format(name, self.config.tag(), ext))