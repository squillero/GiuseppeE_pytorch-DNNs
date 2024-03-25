import torch
import torchvision
import os

BATCH_SIZE = 1

def main():
    model = torchvision.models.mnasnet1_3(weights='IMAGENET1K_V1')

    dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)

    torch.onnx.export(model, dummy_input, "./nvbitPERfi/test-apps/pytorch-DNNs/TensorRT_CNNs/mnas_training/onnx_ckpt/resnet50_onnx_model.onnx", verbose=False)

    TRT_model_name = 'mnasnet_engine.trt'

    cmd = f"trtexec --onnx=resnet50_onnx_model.onnx --saveEngine={TRT_model_name}"
    
    os.system(cmd)
    # remember that when you will use the quantization, also the input must be quantized on the same format. 

if __name__=='__main__':
    main()