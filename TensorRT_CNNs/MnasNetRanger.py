import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.common_types as ct
import torchvision
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import os, sys
import argparse
import nvbitfi_DNN as nvbitDNN
import h5py



def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('--golden', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    parser.add_argument('-w','--workers', required=False, type=int, default=4, help='golden')
    parser.add_argument('-ims','--num_images', required=False, type=int, default=4, help='golden')
    return parser





def main(args):

    path = os.path.dirname(__file__)
    # os.environ["CUDA_VISIBLE_DEVICES"]=""

    # Define relevant variables for the ML task
    batch_size = args.batch_size #512 
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10
    model = torchvision.models.mnasnet0_5()
    model.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout for regularization (optional)
            nn.Linear(1280, num_classes)  # Adjust the input size to match the MNASNet0.5 output features
        )

    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # device = 'cpu'

    model.layers[2]= torch.nn.Hardtanh(min_val=0, max_val=75.5877, inplace=True)
    model.layers[5]= torch.nn.Hardtanh(min_val=0, max_val=92.8986, inplace=True)

    model.layers[8][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=131.5047, inplace=True)
    model.layers[8][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=90.9159, inplace=True)
    model.layers[8][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=63.1861, inplace=True)
    model.layers[8][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=98.8398, inplace=True)
    model.layers[8][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=56.8219, inplace=True)
    model.layers[8][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=76.0987, inplace=True)

    model.layers[9][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=120.0410, inplace=True)
    model.layers[9][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=75.0272, inplace=True)
    model.layers[9][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=38.6695, inplace=True)
    model.layers[9][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=51.2005, inplace=True)
    model.layers[9][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=42.8507, inplace=True)
    model.layers[9][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=71.6337, inplace=True)

    model.layers[10][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=57.3215, inplace=True)
    model.layers[10][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=65.0523, inplace=True)
    model.layers[10][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=27.1506, inplace=True)
    model.layers[10][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=42.3839, inplace=True)
    model.layers[10][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=29.0734, inplace=True)
    model.layers[10][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=48.3887, inplace=True)

    model.layers[11][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=35.3826, inplace=True)
    model.layers[11][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=48.6344, inplace=True)
    model.layers[11][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=21.7015, inplace=True)
    model.layers[11][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=46.8563, inplace=True)

    model.layers[12][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=34.9653, inplace=True)
    model.layers[12][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=51.7370, inplace=True)
    model.layers[12][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=30.0171, inplace=True)
    model.layers[12][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=62.6569, inplace=True)
    model.layers[12][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=23.5628, inplace=True)
    model.layers[12][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=49.5519, inplace=True)
    model.layers[12][3].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=26.1738, inplace=True)
    model.layers[12][3].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=51.1890, inplace=True)

    model.layers[13][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=23.2533, inplace=True)
    model.layers[13][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=44.8555, inplace=True)

    model.layers[16]= torch.nn.Hardtanh(min_val=0, max_val=10.4973, inplace=True)

    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.CIFAR10(
        root='~/dataset/cifar10',
        train=True,
        transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomCrop((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize(mean= (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.RandomRotation(degrees=(-20,20)),
        ]),
        download=True,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='~/dataset/cifar10',
        train=False,
        transform = transforms.Compose([
            transforms.Resize((70, 70)),        
            transforms.CenterCrop((64, 64)),            
            transforms.ToTensor(),                
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    print(args.golden)

    if args.golden:
        # torch.backends.cudnn.allow_tf32=True
        # torch.backends.cudnn.enabled=False

        # print(torch.backends.cuda.matmul.allow_tf32)
        # print(torch.backends.cudnn.allow_tf32)
        
        state_dict = torch.load(
            os.path.join(path, "./checkpoint/mnasnet.pth"), map_location=torch.device("cpu")
        )['state_dict']
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(model)


        t = time.time()
        tot_imgs=0
        gacc1=0
        gacc5=0
        Inputs =[]
        Labels = []
        Output = []
        dummy_input = None
        with torch.no_grad():
            for batch, (images, labels) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if batch == 0: dummy_input = images
                Inputs.append(images.detach().cpu())
                Labels.append(labels.detach().cpu())
                # if(labels[0].item()==0):
                outputs = model(images)
                Output.append(outputs.detach().cpu())
                # sorted, indices=torch.sort(outputs.data)
                pred, clas=outputs.cpu().topk(5,1,True,True)
                
                clas = clas.t()
                Res = clas.eq(labels[None].cpu())
                acc1 = Res[:1].sum(dim=0,dtype=torch.float32)
                acc5 = Res[:5].sum(dim=0,dtype=torch.float32)
                tot_imgs+=batch_size
                gacc1 += Res[:1].flatten().sum(dtype=torch.float32)
                gacc5 += Res[:5].flatten().sum(dtype=torch.float32)
                if batch*batch_size+batch_size>=args.num_images:
                    break
            elapsed = time.time() - t
            print(
                "Accuracy of the network on the {} test images: acc1 {} % acc5 {} % in {} sec".format(
                    tot_imgs, 100 * gacc1 / tot_imgs,  100 * gacc5 / tot_imgs, elapsed
                )
            )

        # Embeddings.extract_embeddings_target_layer()


        currentPath = os.path.dirname(__file__)
        currentFileName = os.path.basename(__file__).split('.')[0]
        directory = os.path.join(currentPath,"DNNs",currentFileName)

        os.system(f"mkdir -p {directory}")

        embeddings_input = (
                torch.cat(Inputs).cpu().numpy()
            )
        
        embeddings_label = (
                torch.cat(Labels).cpu().numpy()
            )
               
        log_path_file = os.path.join(
                directory, f"Inputs_DNN.h5"
            )
        
        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "inputs", data=embeddings_input, compression="gzip"
            )
            hf.create_dataset(
                "labels", data=embeddings_label, compression="gzip"
            )

        embeddings_output = (torch.cat(Output).cpu().numpy())
        print(embeddings_output)
        log_path_file = os.path.join(
                directory, f"Outputs_DNN.h5"
            )
        
        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "outputs", data=embeddings_output, compression="gzip"
            )



        onnx_model_name = os.path.join(directory,f"{currentFileName}_pytorch.onnx")

        torch.onnx.export(model, dummy_input, onnx_model_name, verbose=False)



if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

