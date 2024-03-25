import torch
import torchvision
from torchvision.datasets import CIFAR10 
from torch.utils.data import DataLoader
import torchvision.transforms as trsf
import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    # parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--model_name', required=True, help='name of the target model as it is saved in torchvision library')
    return parser


def get_transformer(mode: str, input_size = (32,32)):
    if mode == 'train':
        transform = trsf.Compose([
            trsf.Resize((70, 70)),
            trsf.RandomCrop((40, 40)),
            trsf.ToTensor(),
            trsf.Normalize(mean= (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            trsf.RandomRotation(degrees=(-20,20)),
        ])
    elif mode == 'test':
        transform = trsf.Compose([
            trsf.Resize((70, 70)),        
            trsf.CenterCrop((64, 64)),            
            trsf.ToTensor(),                
            trsf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    return transform

def adaptable_model_exploration(module, act_counter, hooks, prev_layer):
        # hooks = list()
        for m in module.named_children():
            # sequential_dict = torch.nn.ModuleDict()
            if not isinstance(m[1], torch.nn.ReLU):
                adaptable_model_exploration(m[1], act_counter, hooks, None)
            elif isinstance(m[1], torch.nn.BatchNorm2d):
                prev_layer = module._modules[m[0]]
                adaptable_model_exploration(m[1], act_counter, hooks, prev_layer=prev_layer)
            else:
                # current_max = max(hooks[act_counter].maxs)
                module._modules[m[0]] = prev_layer

                act_counter += 1
        return 

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
    model = models_dict[args.model_name]()
    # state_dict = torch.load('script/teacher_training/ckpt/mobilenet_cifar_SGD/step_3_gamma_0.99_start_lr_0.15_weight_decay_6e-05/76.116470_399epoch.pth')
    # model.load_state_dict(state_dict['state_dict'])

    # Iteratively set the hooks for the activation functions' ouput
    hooks = model_exploration(model, [])

    # DATASETS
    transformer = get_transformer('train')
    train_set = CIFAR10('~/dataset/cifar100', transform=transformer, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size = 128, shuffle=True, pin_memory=True)

    model.to(device='cuda')

    # Perform an inference step to actually catch the outputs from the training set
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device='cuda')
        input = input.to(device='cuda')

        with torch.no_grad():
            # compute output
            output = model(input)


    # Replace the ReLU activation functions with HardTanH
    act_counter = 0

    adaptable_model_exploration(model, act_counter, hooks)
    print(model)

    # repeat training strategy


if __name__=="__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

