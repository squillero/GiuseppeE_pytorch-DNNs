import os
import h5py
import numpy as np
import nvbitfi_DNN_TRT as nvbitfi_DNN
import argparse
import json


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-t','--type', required=False, type=str, help='golden')
    parser.add_argument('-n','--model_name', required=False, type=str, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, help='golden')
    parser.add_argument('-sz','--shape',required=False, nargs='+', type=int ,help="shape of the output layer")
    parser.add_argument('-onnx','--onnx', required=False, action='store_true', help='golden')
    parser.add_argument('-trt','--run_trt', required=False, action='store_true', help='golden')
    return parser

def main(args):
    APPS_DICTIONAY = {}
    current_path = os.getcwd()

    layer_type = "conv2d"
    model_name = "AlexNet"
    for i in range(5):
        os.system(f"python {model_name}.py --golden 1 -ln {i} -bs 1")
        os.system(f"python Run_Layer.py -t {layer_type} -n {model_name} -ln {i} -bs 1 -onnx")

        path = f"{layer_type}/{model_name}-ln{i}"
        log_path_file = os.path.join(
            current_path, path,
            f"Output_layer.h5",
        )
        with h5py.File(log_path_file, "r") as hf:
            Output_dataset = np.array(hf["layer_output"])

        shape = list(Output_dataset[0].reshape(1,*(Output_dataset[0].shape) ).shape)
        
        shape_str=""
        for idx in shape:
            shape_str+=f"{idx} "
        cmd = f"PRELOAD_FLAG= APP_DIR=. BIN_DIR=. APP_BIN=Run_Layer.py ./run.sh -t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        print(cmd)
        os.system(cmd)
        APPS_DICTIONAY[f"{layer_type}-{model_name}-ln{i}"] =[
            f"{current_path}",
            "Run_Layer.py",
            f"{current_path}",
            10,
            f"-t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        ]
     

    layer_type = "conv2d"
    model_name = "LeNet"
    for i in range(2):
        os.system(f"python {model_name}.py --golden 1 -ln {i} -bs 1")
        os.system(f"python Run_Layer.py -t {layer_type} -n {model_name} -ln {i} -bs 1 -onnx")

        path = f"{layer_type}/{model_name}-ln{i}"
        log_path_file = os.path.join(
            current_path, path,
            f"Output_layer.h5",
        )
        with h5py.File(log_path_file, "r") as hf:
            Output_dataset = np.array(hf["layer_output"])

        shape = list(Output_dataset[0].reshape(1,*(Output_dataset[0].shape) ).shape)
        
        shape_str=""
        for idx in shape:
            shape_str+=f"{idx} "
        cmd = f"PRELOAD_FLAG= APP_DIR=. BIN_DIR=. APP_BIN=Run_Layer.py ./run.sh -t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        print(cmd)
        os.system(cmd)
        APPS_DICTIONAY[f"{layer_type}-{model_name}-ln{i}"] =[
            f"{current_path}",
            "Run_Layer.py",
            f"{current_path}",
            10,
            f"-t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        ]

    layer_type = "conv2d"
    model_name = "MobileNetv3"
    for i in range(7):
        os.system(f"python {model_name}.py --golden 1 -ln {i} -bs 1")
        os.system(f"python Run_Layer.py -t {layer_type} -n {model_name} -ln {i} -bs 1 -onnx")

        path = f"{layer_type}/{model_name}-ln{i}"
        log_path_file = os.path.join(
            current_path, path,
            f"Output_layer.h5",
        )
        with h5py.File(log_path_file, "r") as hf:
            Output_dataset = np.array(hf["layer_output"])

        shape = list(Output_dataset[0].reshape(1,*(Output_dataset[0].shape) ).shape)
        
        shape_str=""
        for idx in shape:
            shape_str+=f"{idx} "
        cmd = f"PRELOAD_FLAG= APP_DIR=. BIN_DIR=. APP_BIN=Run_Layer.py ./run.sh -t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        print(cmd)
        os.system(cmd)
        APPS_DICTIONAY[f"{layer_type}-{model_name}-ln{i}"] =[
            f"{current_path}",
            "Run_Layer.py",
            f"{current_path}",
            10,
            f"-t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        ]

    layer_type = "conv2d"
    model_name = "ResNet50"
    for i in range(5):
        os.system(f"python {model_name}.py --golden 1 -ln {i} -bs 1")
        os.system(f"python Run_Layer.py -t {layer_type} -n {model_name} -ln {i} -bs 1 -onnx")

        path = f"{layer_type}/{model_name}-ln{i}"
        log_path_file = os.path.join(
            current_path, path,
            f"Output_layer.h5",
        )
        with h5py.File(log_path_file, "r") as hf:
            Output_dataset = np.array(hf["layer_output"])

        shape = list(Output_dataset[0].reshape(1,*(Output_dataset[0].shape) ).shape)
        
        shape_str=""
        for idx in shape:
            shape_str+=f"{idx} "
        cmd = f"PRELOAD_FLAG= APP_DIR=. BIN_DIR=. APP_BIN=Run_Layer.py ./run.sh -t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        print(cmd)
        os.system(cmd)
        APPS_DICTIONAY[f"{layer_type}-{model_name}-ln{i}"] =[
            f"{current_path}",
            "Run_Layer.py",
            f"{current_path}",
            10,
            f"-t {layer_type} -n {model_name} -ln {i} -bs 1 -trt -sz {shape_str}"
        ]

    print(APPS_DICTIONAY)
    with open('DNN_WORKLOADS.json', 'w') as outfile:
        json.dump(APPS_DICTIONAY,outfile)


if __name__=="__main__":
    argparser = get_argparser()
    args, unknown = argparser.parse_known_args()
    main(args)