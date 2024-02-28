# PyTorch-DNNs

This repository contains the workflow to implement several DNNs using TensorRT for reliability evaluations of permanent faults on GPU devices. This repository works along the reliability evaluation framework [**nvbitPERfi**](https://github.com/divadnauj-GB/nvbitPERfi/tree/Dev). The repository contains several folders; the main one is [**TensorRT_CNNs**](https://github.com/divadnauj-GB/pytorch-DNNs/tree/main/TensorRT_CNNs). 
The main script is [config_layers.py](https://github.com/divadnauj-GB/pytorch-DNNs/blob/main/TensorRT_CNNs/config_layers.py). 

In order to use this framework, you have to follow these steps:
1. Download the ImageNet dataset in your home directory as ```~/dataset/ilsvrc2012```
2. Install or verify that your machine has TensorRT installed. Preferably, use tensor in a conda environment to avoid problems with other Python environments and packages. Further details are presented in [README.md](https://github.com/divadnauj-GB/pytorch-DNNs/blob/main/TensorRT_CNNs/README.md)
3. Execute the main script **config_layers.py** by typing the command ```python config_layers.py```
4. if the previous step runs successfully, then you should have a new file with the configurations of the models as presented in [DNN_WORKLOADS.json](https://github.com/divadnauj-GB/pytorch-DNNs/blob/main/TensorRT_CNNs/DNN_WORKLOADS.json) and new folder called **DNNs** which contain one folder per DNN model. In this example, you should get four models: LeNet, AlexNet, MobileNetv3, and ResNet50. you can edit the scripts to add further DNN models. The content of the file DNN_WORKLOADS.json has to be included in the [real_workloads_parameters.py](https://github.com/divadnauj-GB/nvbitPERfi/blob/Dev/scripts/real_workloads_parameters.py) file of **nvbitPERfi**
5. run the following command to execute the DNN model using the TensorRT implementations of each DNN model. For example, to check the Lenet model, you should type this command in your terminal:

   ```PRELOAD_FLAG= GOLDEN_FLAG=1 APP_DIR=. BIN_DIR=. APP_BIN=DNN_TRT.py ./run.sh -t DNNs -n LeNet -bs 1 -trt -sz 1 10```
