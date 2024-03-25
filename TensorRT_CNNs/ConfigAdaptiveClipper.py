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
    parser.add_argument('--path', required=True, help='pretrained model file path')
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

    if os.path.exists(args.path):
        print('*****************************************************')
        pth_file = torch.load(args.path)
        state_dict = pth_file['state_dict']
        model.load_state_dict(state_dict=state_dict)


    # state_dict = torch.load('script/teacher_training/ckpt/mobilenet_cifar_SGD/step_3_gamma_0.99_start_lr_0.15_weight_decay_6e-05/76.116470_399epoch.pth')
    # model.load_state_dict(state_dict['state_dict'])

    # Iteratively set the hooks for the activation functions' ouput
    # hooks = model_exploration(model, [])
    hook1 = IntermediateOutputHook(model.layers[2])
    hook2 = IntermediateOutputHook(model.layers[5])

    hook3 = IntermediateOutputHook(model.layers[8][0].layers[2])
    hook4 = IntermediateOutputHook(model.layers[8][0].layers[5])
    hook5 = IntermediateOutputHook(model.layers[8][1].layers[2])
    hook6 = IntermediateOutputHook(model.layers[8][1].layers[5])
    hook7 = IntermediateOutputHook(model.layers[8][2].layers[2])
    hook8 = IntermediateOutputHook(model.layers[8][2].layers[5])

    hook9 = IntermediateOutputHook(model.layers[9][0].layers[2])
    hook10 = IntermediateOutputHook(model.layers[9][0].layers[5])
    hook11 = IntermediateOutputHook(model.layers[9][1].layers[2])
    hook12 = IntermediateOutputHook(model.layers[9][1].layers[5])
    hook13 = IntermediateOutputHook(model.layers[9][2].layers[2])
    hook14 = IntermediateOutputHook(model.layers[9][2].layers[5])

    hook15 = IntermediateOutputHook(model.layers[10][0].layers[2])
    hook16 = IntermediateOutputHook(model.layers[10][0].layers[5])
    hook17 = IntermediateOutputHook(model.layers[10][1].layers[2])
    hook18 = IntermediateOutputHook(model.layers[10][1].layers[5])
    hook19 = IntermediateOutputHook(model.layers[10][2].layers[2])
    hook20 = IntermediateOutputHook(model.layers[10][2].layers[5])

    hook21 = IntermediateOutputHook(model.layers[11][0].layers[2])
    hook22 = IntermediateOutputHook(model.layers[11][0].layers[5])
    hook23 = IntermediateOutputHook(model.layers[11][1].layers[2])
    hook24 = IntermediateOutputHook(model.layers[11][1].layers[5])

    hook25 = IntermediateOutputHook(model.layers[12][0].layers[2])
    hook26 = IntermediateOutputHook(model.layers[12][0].layers[5])
    hook27 = IntermediateOutputHook(model.layers[12][1].layers[2])
    hook28 = IntermediateOutputHook(model.layers[12][1].layers[5])
    hook29 = IntermediateOutputHook(model.layers[12][2].layers[2])
    hook30 = IntermediateOutputHook(model.layers[12][2].layers[5])
    hook31 = IntermediateOutputHook(model.layers[12][3].layers[2])
    hook32 = IntermediateOutputHook(model.layers[12][3].layers[5])

    hook33 = IntermediateOutputHook(model.layers[13][0].layers[2])
    hook34 = IntermediateOutputHook(model.layers[13][0].layers[5])

    hook35 = IntermediateOutputHook(model.layers[16])

    hooks = []
    



    # DATASETS
    transformer = get_transformer('train')
    train_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size = 128, shuffle=True, pin_memory=True)

    transformer = get_transformer('test')
    val_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True, train=False)
    val_loader = DataLoader(dataset=val_set, batch_size = 128, shuffle=True, pin_memory=True)



    model.to(device='cuda')

    # Perform an inference step to actually catch the outputs from the training set
    for i, (input, target) in tqdm(enumerate(train_loader)):
        target = target.to(device='cuda')
        input = input.to(device='cuda')

        with torch.no_grad():
            # compute output
            output = model(input)

    # Replace the ReLU activation functions with HardTanH
    # act_counter = 0

    # adaptable_model_exploration(model, act_counter, hooks, 0)
    # print(model)
    
    hooks.append(max(hook1.maxs))
    hooks.append(max(hook2.maxs))
    hooks.append(max(hook3.maxs))
    hooks.append(max(hook4.maxs))
    hooks.append(max(hook5.maxs))
    hooks.append(max(hook6.maxs))
    hooks.append(max(hook7.maxs))
    hooks.append(max(hook8.maxs))
    hooks.append(max(hook9.maxs))
    hooks.append(max(hook10.maxs))
    hooks.append(max(hook11.maxs))
    hooks.append(max(hook12.maxs))
    hooks.append(max(hook13.maxs))
    hooks.append(max(hook14.maxs))
    hooks.append(max(hook15.maxs))
    hooks.append(max(hook16.maxs))
    hooks.append(max(hook17.maxs))
    hooks.append(max(hook18.maxs))
    hooks.append(max(hook19.maxs))
    hooks.append(max(hook20.maxs))
    hooks.append(max(hook21.maxs))
    hooks.append(max(hook22.maxs))
    hooks.append(max(hook23.maxs))
    hooks.append(max(hook24.maxs))
    hooks.append(max(hook25.maxs))
    hooks.append(max(hook26.maxs))
    hooks.append(max(hook27.maxs))
    hooks.append(max(hook28.maxs))
    hooks.append(max(hook29.maxs))
    hooks.append(max(hook30.maxs))
    hooks.append(max(hook31.maxs))
    hooks.append(max(hook32.maxs))
    hooks.append(max(hook33.maxs))
    hooks.append(max(hook34.maxs))
    hooks.append(max(hook35.maxs))

    print(hooks)
    
    model = torchvision.models.mnasnet0_5(pretrained=True, progress=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),  # Add dropout for regularization (optional)
        torch.nn.Linear(1280, 10)  # Adjust the input size to match the MNASNet0.5 output features
    )

    model.layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook1.maxs), decimals=3), inplace=True)
    model.layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook2.maxs), decimals=3), inplace=True)

    model.layers[8][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook3.maxs), decimals=3), inplace=True)
    model.layers[8][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook4.maxs), decimals=3), inplace=True)
    model.layers[8][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook5.maxs), decimals=3), inplace=True)
    model.layers[8][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook6.maxs), decimals=3), inplace=True)
    model.layers[8][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook7.maxs), decimals=3), inplace=True)
    model.layers[8][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook8.maxs), decimals=3), inplace=True)

    model.layers[9][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook9.maxs), decimals=3), inplace=True)
    model.layers[9][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook10.maxs), decimals=3), inplace=True)
    model.layers[9][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook11.maxs), decimals=3), inplace=True)
    model.layers[9][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook12.maxs), decimals=3), inplace=True)
    model.layers[9][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook13.maxs), decimals=3), inplace=True)
    model.layers[9][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook14.maxs), decimals=3), inplace=True)

    model.layers[10][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook15.maxs), decimals=3), inplace=True)
    model.layers[10][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook16.maxs), decimals=3), inplace=True)
    model.layers[10][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook17.maxs), decimals=3), inplace=True)
    model.layers[10][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook18.maxs), decimals=3), inplace=True)
    model.layers[10][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook19.maxs), decimals=3), inplace=True)
    model.layers[10][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook20.maxs), decimals=3), inplace=True)

    model.layers[11][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook21.maxs), decimals=3), inplace=True)
    model.layers[11][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook22.maxs), decimals=3), inplace=True)
    model.layers[11][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook23.maxs), decimals=3), inplace=True)
    model.layers[11][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook24.maxs), decimals=3), inplace=True)

    model.layers[12][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook25.maxs), decimals=3), inplace=True)
    model.layers[12][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook26.maxs), decimals=3), inplace=True)
    model.layers[12][1].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook27.maxs), decimals=3), inplace=True)
    model.layers[12][1].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook28.maxs), decimals=3), inplace=True)
    model.layers[12][2].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook29.maxs), decimals=3), inplace=True)
    model.layers[12][2].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook30.maxs), decimals=3), inplace=True)
    model.layers[12][3].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook31.maxs), decimals=3), inplace=True)
    model.layers[12][3].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook32.maxs), decimals=3), inplace=True)

    model.layers[13][0].layers[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook33.maxs), decimals=3), inplace=True)
    model.layers[13][0].layers[5]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook34.maxs), decimals=3), inplace=True)

    model.layers[16]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(max(hook35.maxs), decimals=3), inplace=True)

    optimizer = torch.optim.AdamW(model.parameters(), 0.01,
                                    weight_decay=6e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    best_prec1 = 0.0
    train_loss_avg=0.0
    val_loss_avg=0.0
    val_prec1_avg=0.0

    writer = SummaryWriter("./nvbitPERfi/test-apps/pytorch-DNNs/TensorRT_CNNs/checkpoint/Adaptive_clipper_ckpt")

    for epoch in tqdm(range(100)):
            print(f'epoch: {epoch}')
            train_loss_avg, train_prec1_avg, train_prec5_avg = train(model, criterion=criterion, optimizer = optimizer, dataloader=train_loader, epoch=epoch, writer=writer)

            val_loss_avg, val_prec1_avg, val_prec5_avg = validate(model, val_loader=val_loader, criterion=criterion)
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
                }, is_best=is_best, checkpoint='/home/g.esposito/nvbit_release/tools/nvbitPERfi/test-apps/pytorch-DNNs/TensorRT_CNNs/checkpoint/Adaptive_clipper_ckpt', filename='{:3f}_{}epoch.pth'.format(best_prec1, epoch))
    # repeat training strategy
    print(model)
    


if __name__=="__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

