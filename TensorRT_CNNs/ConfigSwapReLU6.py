import torch
import torchvision
from torchvision.datasets import CIFAR10 
from torch.utils.data import DataLoader
import torchvision.transforms as trsf
import argparse
import os
from tqdm import tqdm
from mnas_training.NasNet_mobile_finetuning import train, validate, save_checkpoint, get_transformer
from torch.utils.tensorboard import SummaryWriter

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    # parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--model_name', required=True, help='name of the target model as it is saved in torchvision library')
    parser.add_argument('--path', required=False, help='pretrained model file path')
    return parser



def adaptable_model_exploration(module, act_counter, hooks, counter):
        # hooks = list()
        counter+=1
        for m in module.named_children():
            # sequential_dict = torch.nn.ModuleDict()
            if isinstance(m[1], torch.nn.ReLU):
                current_max = max(hooks[act_counter].maxs)
                module._modules[m[0]] = torch.nn.Hardtanh(min_val=0, max_val=current_max, inplace=True)
                act_counter += 1
            elif isinstance(m[1], torch.nn.Sequential):
                adaptable_model_exploration(m[1], act_counter, hooks, counter)
            else:
                continue
        print(counter)

def model_exploration(module, hooks):
        for m in module.named_children():
            if not isinstance(m[1], torch.nn.ReLU):
                hooks = model_exploration(m[1], hooks)
            else:
                hooks.append(IntermediateOutputHook(m[1]))
        return hooks


class IntermediateOutputHook:
    def __init__(self, module):
        self.maxs = []
        print(module)
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        max_ = torch.max(output)
        self.maxs.append(max_)

    def remove(self):
        self.hook.remove()

# Define the model
# model = MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=10)
def main(args):
    models_dict = torchvision.models.__dict__
    # print(models_dict.keys())
    model = models_dict[args.model_name]()

    model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),  # Add dropout for regularization (optional)
            torch.nn.Linear(1280, 10)  # Adjust the input size to match the MNASNet0.5 output features
        )

    if args.path is not None:
        if os.path.exists(args.path):
            print('*****************************************************')
            pth_file = torch.load(args.path)
            state_dict = pth_file['state_dict']
            model.load_state_dict(state_dict=state_dict)

    



    # DATASETS
    transformer = get_transformer('train')
    train_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size = 128, shuffle=True, pin_memory=True)

    transformer = get_transformer('test')
    val_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True, train=False)
    val_loader = DataLoader(dataset=val_set, batch_size = 128, shuffle=True, pin_memory=True)



    model.to(device='cuda')

    
    model = torchvision.models.mnasnet0_5(pretrained=True, progress=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),  # Add dropout for regularization (optional)
        torch.nn.Linear(1280, 10)  # Adjust the input size to match the MNASNet0.5 output features
    )

    model.layers[2]= model.layers[1]
    model.layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[5]= model.layers[4]
    model.layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[8][0].layers[2]= model.layers[8][0].layers[1]
    model.layers[8][0].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[8][0].layers[5]= model.layers[8][0].layers[4]
    model.layers[8][0].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[8][1].layers[2]= model.layers[8][1].layers[1]
    model.layers[8][1].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[8][1].layers[5]= model.layers[8][1].layers[4]
    model.layers[8][1].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[8][2].layers[2]= model.layers[8][2].layers[1]
    model.layers[8][2].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[8][2].layers[5]= model.layers[8][2].layers[4]
    model.layers[8][2].layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[9][0].layers[2]= model.layers[9][0].layers[1]
    model.layers[9][0].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[9][0].layers[5]= model.layers[9][0].layers[4]
    model.layers[9][0].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[9][1].layers[2]= model.layers[9][1].layers[1]
    model.layers[9][1].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[9][1].layers[5]= model.layers[9][1].layers[4]
    model.layers[9][1].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[9][2].layers[2]= model.layers[9][2].layers[1]
    model.layers[9][2].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[9][2].layers[5]= model.layers[9][2].layers[4]
    model.layers[9][2].layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[10][0].layers[2]= model.layers[10][0].layers[1]
    model.layers[10][0].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[10][0].layers[5]= model.layers[10][0].layers[4]
    model.layers[10][0].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[10][1].layers[2]= model.layers[10][1].layers[1]
    model.layers[10][1].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[10][1].layers[5]= model.layers[10][1].layers[4]
    model.layers[10][1].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[10][2].layers[2]= model.layers[10][2].layers[1]
    model.layers[10][2].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[10][2].layers[5]= model.layers[10][2].layers[4]
    model.layers[10][2].layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[11][0].layers[2]= model.layers[11][0].layers[1]
    model.layers[11][0].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[11][0].layers[5]= model.layers[11][0].layers[4]
    model.layers[11][0].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[11][1].layers[2]= model.layers[11][1].layers[1]
    model.layers[11][1].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[11][1].layers[5]= model.layers[11][1].layers[4]
    model.layers[11][1].layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[12][0].layers[2]= model.layers[12][0].layers[1]
    model.layers[12][0].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[12][0].layers[5]= model.layers[12][0].layers[4]
    model.layers[12][0].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[12][1].layers[2]= model.layers[12][1].layers[1]
    model.layers[12][1].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[12][1].layers[5]= model.layers[12][1].layers[4]
    model.layers[12][1].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[12][2].layers[2]= model.layers[12][2].layers[1]
    model.layers[12][2].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[12][2].layers[5]= model.layers[12][2].layers[4]
    model.layers[12][2].layers[4] = torch.nn.ReLU6(inplace=True)
    model.layers[12][3].layers[2]= model.layers[12][3].layers[1]
    model.layers[12][3].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[12][3].layers[5]= model.layers[12][3].layers[4]
    model.layers[12][3].layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[13][0].layers[2]= model.layers[13][0].layers[1]
    model.layers[13][0].layers[1] = torch.nn.ReLU6(inplace=True)
    model.layers[13][0].layers[5]= model.layers[13][0].layers[4]
    model.layers[13][0].layers[4] = torch.nn.ReLU6(inplace=True)

    model.layers[16]= model.layers[15]
    model.layers[15]= torch.nn.ReLU6(inplace=True)

    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                    weight_decay=6e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    print(model)

    best_prec1 = 0.0
    train_loss_avg=0.0
    val_loss_avg=0.0
    val_prec1_avg=0.0

    writer = SummaryWriter("./nvbitPERfi/test-apps/pytorch-DNNs/TensorRT_CNNs/checkpoint/Adaptive_clipper_ckpt")

    for epoch in tqdm(range(100)):
            print(f'epoch: {epoch}')
            train_loss_avg, train_prec1_avg, train_prec5_avg, counter = train(model, criterion=criterion, optimizer = optimizer, dataloader=train_loader, epoch=epoch, writer=writer)

            val_loss_avg, val_prec1_avg, val_prec5_avg, _ = validate(model, val_loader=val_loader, criterion=criterion, softmax=None)
            print(f'val_loss_avg: {val_loss_avg}, val_prec1_avg: {val_prec1_avg}, val_prec5_avg: {val_prec5_avg}')
            scheduler.step()


            is_best = val_prec1_avg > best_prec1
            best_prec1 = max(val_prec1_avg, best_prec1)
            is_best = val_prec1_avg > best_prec1
            best_prec1 = max(val_prec1_avg, best_prec1)

            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, checkpoint='/home/g.esposito/nvbit_release/tools/nvbitPERfi/test-apps/pytorch-DNNs/TensorRT_CNNs/checkpoint/SwapReLU6_ckpt', filename='{:3f}_{}epoch.pth'.format(best_prec1, epoch))
    # repeat training strategy
    print(model)
    


if __name__=="__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

