import torch
import torchvision
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse 
import numpy as np
import os, time


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('--golden', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    parser.add_argument('-w','--workers', required=False, type=int, default=4, help='golden')
    parser.add_argument('-ims','--num_images', required=False, type=int, default=4, help='golden')
    return parser


def main(args):

    BATCH_SIZE = args.batch_size
    target_dtype = np.float32
    path = os.path.dirname(__file__)
    currentFileName = os.path.basename(__file__).split('.')[0].split('_')[0]

    train_transforms = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomCrop((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize(mean= (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.RandomRotation(degrees=(-20,20)),
        ])

    test_transform = transforms.Compose([
            transforms.Resize((70, 70)),        
            transforms.CenterCrop((64, 64)),            
            transforms.ToTensor(),                
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Define relevant variables for the ML task
    batch_size = args.batch_size #512 


    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # device = 'cpu'
    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.CIFAR10(
        root="~/dataset/cifar10",
        transform=train_transforms
    )

    test_dataset = train_dataset = torchvision.datasets.CIFAR10(
        root="~/dataset/cifar10",
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    #print(args.golden)

    TRT_model_name = "mnasnet_pytorch.rtr"
    Num_outouts = 1000
    
    with open(os.path.join(path, "DNNs", currentFileName, TRT_model_name), "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        output = np.empty([BATCH_SIZE, Num_outouts], dtype = target_dtype) 
        # allocate device memory
        for batch, (images, labels) in enumerate(test_loader):
            sample_images = np.array(images, dtype=np.float32)
            break

        d_input = cuda.mem_alloc(1 * sample_images.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()

        t = time.time()
        tot_imgs=0
        gacc1=0
        gacc5=0
        dummy_input=None
        for batch, (images, labels) in enumerate(test_loader):
            images = np.array(images, dtype=np.float32)

            cuda.memcpy_htod_async(d_input, images, stream)
            # execute model
            context.execute_async_v2(bindings, stream.handle, None)
            # transfer predictions back
            cuda.memcpy_dtoh_async(output, d_output, stream)
            # syncronize threads
            stream.synchronize()
            outputs = torch.from_numpy(output)
            pred, clas=outputs.cpu().topk(5,1,True,True)
            clas = clas.t()
            pred = pred.t()
            size = pred.shape
            
            # for idx,label in enumerate(labels):
            #     for pred_top in range(size[0]):
            #         print(f"{batch*BATCH_SIZE+idx}; {pred_top}; {label}; {clas[pred_top][idx]}; {pred[pred_top][idx]}")
            Res = clas.eq(labels[None].cpu())

            acc1 = Res[:1].sum(dim=0,dtype=torch.float32)
            acc5 = Res[:5].sum(dim=0,dtype=torch.float32)            
            gacc1 += Res[:1].flatten().sum(dtype=torch.float32)
            gacc5 += Res[:5].flatten().sum(dtype=torch.float32)
            tot_imgs+=BATCH_SIZE
            if batch*BATCH_SIZE+BATCH_SIZE>=args.num_images:
                break
        
        elapsed = time.time() - t
        print(
            "Accuracy of the network on the {} test images: acc1 {} % acc5 {} % in {} sec".format(
                tot_imgs, 100 * gacc1 / tot_imgs,  100 * gacc5 / tot_imgs, elapsed
            )
        )


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

