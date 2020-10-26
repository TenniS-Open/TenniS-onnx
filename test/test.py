import tennis_onnx as onnx

def test():
    working_root = "/home/seeta/Documents/models/"
    input_module = working_root + "model_arcface_2020-3-4.tsm"
    input_shape = [(-1, 3, 112, 112)]
    output_module = "output/test.tsm" # working_root + "rknn/arcface.tsm"

    print("=========== Start test ==============")
    exporter = onnx.exporter.ONNXExporter(host_device="gpu")

    exporter.load(input_module, input_shape)
    exporter.export_onnx(output_module, subdir="onnx", export_main=True)

if __name__ == "__main__":
    test()
