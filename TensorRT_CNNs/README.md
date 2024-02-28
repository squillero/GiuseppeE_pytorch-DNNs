This is an instruction manual about how to install TensorRT on Ubuntu 2004 with a specific version of CUDA. I'll put my example here:

1: install CUDA and the Nvidia drivers for that cuda version; make sure they are working properly

2: Install TensorRT using the apt package manager. For this step, follow the instructions for installation presented by Nvidia, but be sure
you are installing the exact version for your cuda version. to do so, type this command:

```
sudo apt-cache policy tensorrt
```
    

After running this command you should get the list of options available to be installed. Here the example I got in my machine.

```
tensorrt:
  Installed: 8.4.3.1-1+cuda11.6
  Candidate: 8.6.1.6-1+cuda12.0
  Version table:
     8.6.1.6-1+cuda12.0 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.6.1.6-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.6.0.12-1+cuda12.0 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.6.0.12-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.5.3.1-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.5.2.2-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.5.1.7-1+cuda11.8 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
 *** 8.4.3.1-1+cuda11.6 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
        100 /var/lib/dpkg/status
     8.4.2.4-1+cuda11.6 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
     8.4.1.5-1+cuda11.6 600
        600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages
```

I have CUDA 11.6; therefore, I install the version of tensorrt accordingly, using the following command:
```
sudo apt-get install tensorrt=8.4.3.1-1+cuda11.6
```

It is possible that an error ride complaining about other dependencies not being installed, so force the installation of every one of those packages, adding the same tensorrt version you want to install

if you can't find the right version using the apt-get command, then you have to follow the tar file installation procedure, which allows you to install tensorrt native on your OS but also allows you to add the python packages on your environment like conda. 

For manually installing the python packages of tensorrt on your conda environment, follow these steps:

1. Download the TensorRT tar file that matches the CPU architecture and CUDA version you are using.
2. Choose where you want to install TensorRT. This tar file will install everything into a subdirectory called TensorRT-8.x.x.x.
3. Unpack the tar file.

```
version="8.x.x.x"
arch=$(uname -m)
cuda="cuda-x.x"
tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz
```
Where:
    8.x.x.x is your TensorRT version
    cuda-x.x is CUDA version 11.8, 11.6 or 12.0
This directory will have sub-directories like lib, include, data, and so on.

```
ls TensorRT-${version}
bin  data  doc  graphsurgeon  include  lib  onnx_graphsurgeon  python  samples  targets  uff
```

4. Add the absolute path to the TensorRT lib directory to the environment variable LD_LIBRARY_PATH:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
```

5. Install the Python TensorRT wheel file (replace cp3x with the desired Python version, for example, cp310 for Python 3.10). (here you can use conda environment or any other you required)

```
cd TensorRT-${version}/python
python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
```

4. Optionally, install the TensorRT lean and dispatch runtime wheel files:

```
python3 -m pip install tensorrt_lean-*-cp3x-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-*-cp3x-none-linux_x86_64.whl
```

5. Install the Python UFF wheel file. This is only required if you plan to use TensorRT with TensorFlow in UFF format.

```
cd TensorRT-${version}/uff
python3 -m pip install uff-0.6.9-py2.py3-none-any.whl
```

6. Check the installation with:

```
which convert-to-uff
```

7. Install the Python graphsurgeon wheel file.

```
cd TensorRT-${version}/graphsurgeon
python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
```

8. Install the Python onnx-graphsurgeon wheel file.

```
cd TensorRT-${version}/onnx_graphsurgeon
    
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
```
