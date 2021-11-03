import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet50

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

mcnn_list = [0, 3, 7, 11, 15, 20] ## This is a bad way to do things

def gradient_quotient(loss, params, eps=1e-5):
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]), params, retain_graph=True, create_graph=True)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach())- 1).abs().sum() for g, p in zip(grad, prod)])
    return out / sum([p.data.nelement() for p in params])

def metainit(model, criterion, x_size, y_size, lr=0.1, momentum=0.9, steps=500, eps=1e-5):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    for i in range(steps):
        input = torch.ones(*x_size).cuda() #torch.Tensor(*x_size).normal_(0, 1).cuda() #
        target = torch.ones(*y_size).cuda() #torch.Tensor(*y_size).normal_(0, 1).cuda() #
        loss = criterion(model(input), target)
        gq = gradient_quotient(loss, list(model.parameters()), eps)
        grad = torch.autograd.grad(gq, params)
        
        for j, (p, g_all) in enumerate(zip(params, grad)):
            norm = p.data.norm().item()
            g = torch.sign((p.data * g_all).sum() / norm)
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            p.data.mul_(new_norm / norm)
        print("%d/GQ = %.2f" % (i, gq.item()))
    
class initialize_scaled_kaiming(object):
    def __init__(self, weightscale, mode, No, scale_bn=False, deep_linear=False):
        self.weightscale = weightscale
        self.mode = mode
        self.No = No
        self.scale_bn = scale_bn
        self.deep_linear = deep_linear

    def initialize(self, module):
        if isinstance(module, nn.Linear) and module.weight.data.shape[0]!=self.No and self.deep_linear is False: # For linear that is followed by ReLU
            print('relu', module.weight.data.shape)
            nn.init.kaiming_normal_(module.weight.data, a=0, mode=self.mode, nonlinearity='relu') # Kaiming init
            module.weight.data = self.weightscale*module.weight.data
            module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, a=0, mode=self.mode, nonlinearity='relu') # Kaiming init
            module.weight.data = self.weightscale*module.weight.data
            try:
                module.bias.data.zero_()
            except:
                pass
        elif isinstance(module, nn.BatchNorm2d):
            if self.scale_bn:
                module.weight.data.fill_(self.weightscale)
            else:
                module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and (module.weight.data.shape[0]==self.No or self.deep_linear is True): # For deep linear or for final layer in conv/relu net
            print('lin', module.weight.data.shape)
            nn.init.kaiming_normal_(module.weight.data, a=0, mode=self.mode, nonlinearity='linear') # Kaiming init
            module.weight.data = self.weightscale*module.weight.data
            try:
                module.bias.data.zero_()
            except:
                pass

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


def get_classifier(settings, No, classifierpath=None):
    
    if settings.dlaas:
        sys.path.insert(0, 'source/models')
    else:
        sys.path.insert(0, '../source/models')
        sys.path.insert(0, '../source/utils')

    from basicmodels import create_fc, create_cnn, Flatten
    from resnet_pytorch_image_classification import resnet_pic
    from wrn import wideresnet_pic, wideresnet_mine
    from resnet import resnet18k_cifar
#    from vit import vit4s
    from model_stitching import get_stitched_classifier, convert_model_to_Seq


    #### Stitch two classifiers #####
    if settings.stitch_models:
        classifier = get_stitched_classifier(settings)
        return classifier
    
    #### Load existing classifier or create new ######
    if classifierpath is not None:
        classifier = torch.load(classifierpath)

        ### Hack for changing output classes
        if settings.classifiername=='mCNN_k_bn_cifar10':
            classifier[-1] = nn.Linear(classifier[-1].in_features, out_features = No, bias=True)
            print(classifier)
                    
    elif settings.classifiername=='FC_linear':
        classifier = create_fc(settings.numinput, settings.depth, settings.numhid, No, nonlin_name=None, batch_norm=settings.add_batchnorm, add_bias=False)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, deep_linear=True)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='FC_linear_metainit':
        classifier = create_fc(settings.numinput, settings.depth, settings.numhid, No, nonlin_name=None, batch_norm=settings.add_batchnorm, add_bias=False)
        P = settings.num_samples
        N = settings.numinput
        No = No
        metainit(classifier, nn.MSELoss(), (P, N), (P, No), lr=0.0001, momentum=0.9, steps=500, eps=1e-5)
    elif settings.classifiername=='FC_relu':
        classifier = create_fc(settings.numinput, settings.depth, settings.numhid, No, nonlin_name='relu', batch_norm=settings.add_batchnorm)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='mCNN_k_cifar10':
        resolution = 32
        d_output = No
        c_in = 3
        c_first = settings.conv_channels
        n_layers = 5
        classifier = create_cnn(resolution, d_output, c_in, c_first, n_layers, batch_norm=False)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='mCNN_k_bn_cifar10':
        resolution = 32
        d_output = No
        c_in = 3
        c_first = settings.conv_channels
        n_layers = 5
        classifier = create_cnn(resolution, d_output, c_in, c_first, n_layers, batch_norm=True)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='mCNN_depth_k_bn_cifar10':
        resolution = 32
        d_output = No
        c_in = 3
        c_first = settings.conv_channels
        n_layers = settings.depth
        classifier = create_cnn(resolution, d_output, c_in, c_first, n_layers, batch_norm=True)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)        
    elif settings.classifiername=='cub_resnet50':
        classifier = torchvision.models.resnet50(pretrained=True)
        d = classifier.fc.in_features
        classifier.fc = nn.Linear(d, No)
    elif settings.classifiername=='resnet50_cifar':
        classifier = resnet50()
        classifier.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        classifier.fc = nn.Linear(2048, No, bias=True)
        classifier.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)        
    elif settings.classifiername=='resnet18k_cifar':
        k = settings.conv_channels
        num_classes = No
        classifier = resnet18k_cifar(k, num_classes)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='resnet110_cifar_notmyinit':
        classifier = preact_resnet110_cifar()
    elif settings.classifiername=='resnet110_cifar_pic':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16
        config['remove_first_relu'] = False
        config['add_last_bn'] = False        
        config['block_type'] = 'basic'
        config['depth'] = 110
        config['preact_stage'] = ['True', 'True', 'True']
        classifier = resnet_pic(config)
    elif settings.classifiername=='resnet110_cifar_pic_myinit':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16
        config['remove_first_relu'] = False
        config['add_last_bn'] = False        
        config['block_type'] = 'basic'
        config['depth'] = 110
        config['preact_stage'] = ['True', 'True', 'True']
        classifier = resnet_pic(config)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='resnet20_cifar_pic_myinit':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16
        config['remove_first_relu'] = False
        config['add_last_bn'] = False        
        config['block_type'] = 'basic'
        config['depth'] = 20
        config['preact_stage'] = ['True', 'True', 'True']
        classifier = resnet_pic(config)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='resnet38_10x_cifar_pic_myinit':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16*10
        config['remove_first_relu'] = False
        config['add_last_bn'] = False        
        config['block_type'] = 'basic'
        config['depth'] = 38
        config['preact_stage'] = ['True', 'True', 'True']
        classifier = resnet_pic(config)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='resnet164_cifar_pic_myinit':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16
        config['remove_first_relu'] = False
        config['add_last_bn'] = False        
        config['block_type'] = 'basic'
        config['depth'] = 164
        config['preact_stage'] = ['True', 'True', 'True']
        classifier = resnet_pic(config)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)                           
    elif settings.classifiername=='wideresnet_pic_myinit':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16
        config['widening_factor'] = 10
        config['drop_rate'] = 0.
        config['depth'] = 28
        classifier = wideresnet_pic(config)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='wideresnet_mine_myinit':
        config = {}
        config['input_shape'] = [1, 3, 32, 32]
        config['n_classes'] = No
        config['base_channels'] = 16
        config['widening_factor'] = 10
        config['drop_rate'] = 0.
        config['depth'] = 28
        classifier = wideresnet_mine(config)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=No, scale_bn=settings.scale_bn, deep_linear=False)
        classifier.apply(class_init.initialize)
    elif settings.classifiername=='vit4s':
        classifier = vit4s()
    elif settings.classifiername=='resnet50_im':
        classifier = resnet50(pretrained=True)
    else:
        raise NotImplementedError

    
    #### Not so nice code for freezing and unfreezing things
    if settings.freeze_layers!=-1:

        if 'resnet' in settings.classifiername:
            classifier = convert_model_to_Seq(classifier, settings.classifiername)

            if isinstance(settings.freeze_layers, list):
                freeze_list = settings.freeze_layers
                reset_list = []
            else:
                freeze_list = range(settings.freeze_layers)
                reset_list = range(settings.freeze_layers, len(classifier))                
            
            for i in freeze_list:
                for param in classifier[i].parameters():
                    param.requires_grad = False

            for i in reset_list:
                try:
                    classifier[i].reset_parameters()
                except:
                    pass

            for i in range(len(classifier)):
                for param in classifier[i].parameters():
                    print(i, param.requires_grad)

    
    if settings.unfreeze_layers!=-1:
        # Freeze everything
        for param in classifier.parameters():
            param.requires_grad = False

        if settings.classifiername=='mCNN_k_bn_cifar10':
            for i in range(20, -1, -1):

                try:
                    classifier[i].reset_parameters()
                except:
                    pass
                
                for param in classifier[i].parameters():
                    param.requires_grad = True
                if i==mcnn_list[-settings.unfreeze_layers]:
                    break
                else:
                    pass

        elif settings.classifiername=='wideresnet_pic_myinit':
            for param in classifier.fc.parameters():
                param.requires_grad = True
            for param in classifier.bn.parameters():
                param.requires_grad = True

            if settings.unfreeze_layers>1:
                for module in list(classifier.stage3.block4.children())[-1]:
                    module.reset_parameters()
                    
                for param in classifier.stage3.block4.parameters():
                    param.requires_grad = True

        ### TRASHHHHHH for a single use case
        elif settings.classifiername=='simclr_resnet':
            for param in classifier[5].parameters():
                param.requires_grad = True
                

        else:
            raise NotImplementedError

    return classifier
