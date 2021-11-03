import torch, sys, os, copy
import torch.nn as nn
import numpy as np

#import timm
#from timm.models.vision_transformer import VisionTransformer
#import clip

class VitPrefix(nn.Module):
    """ Prefix of Vision Transformer
    """

    def __init__(self, base_vit, num_blocks):
        super().__init__()
        self.base = base_vit
        self.num_blocks = num_blocks

    def forward(self, x):
        B = x.shape[0]
        x = self.base.patch_embed(x)

        cls_tokens = self.base.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.base.pos_embed
        x = self.base.pos_drop(x)

        for blk in self.base.blocks[:self.num_blocks]:
            x = blk(x)

        return x

class ClipVitSuffix(nn.Module):
    """ Suffix of CLIP-Vision Transformer
    """

    def __init__(self, base_vit, num_blocks_prefix):
        super().__init__()
        self.base = base_vit
        self.num_blocks = num_blocks_prefix

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.base.transformer.resblocks[self.num_blocks:](x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.base.ln_post(x[:, 0, :])

        if self.base.proj is not None:
            x = x @ self.base.proj

        return x
    
    
def reset_clip_projection(clip_vit, output_dim=10):
    # reset the CLIP-ViT projection head
    width = clip_vit.proj.shape[0] 
    clip_vit.proj = nn.Parameter((width** -0.5) * torch.randn(width, output_dim))


def stitch_vit_clip(numout):
    vit = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)

    clip_model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
    clip_vit = clip_model.visual
    #reset_clip_projection(clip_vit, output_dim=numout) 

    # glue it
    i = 4 # how many blocks to take from the HEAD vit
    j = 4 # how many to skip from the TAIL clip.
    glue = nn.Linear(768, 768, bias=True)
    head = VitPrefix(vit, i)
    tail = ClipVitSuffix(clip_vit, j) # important: reset projection head to have 10 classes

    head.requires_grad_(False)
    glue.requires_grad_(True)
    tail.requires_grad_(False)
    #tail.base.proj.requires_grad_(True) # Train the projection head too (just as a hack)

    joint = nn.Sequential(head, glue, tail)

    return joint, clip_vit

    

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def convert_model_to_Seq(model, classifiername, stitch_model1_path=None):
    if stitch_model1_path is not None and '/n/coxfs01/ybansal/pretrain-rep/CIFAR-10/simclr_resnet18_28w562hk/28w562hk' in stitch_model1_path:
        ### This is only for bottom model - the top layer in this model is random
        module_list = list(model) #[model[0]] + list(model[1]) + list(model[2]) + list(model[3]) + list(model[4]) + [nn.AdaptiveAvgPool2d(output_size=1), Flatten(), model[5]]
    elif 'resnet20' in classifiername or 'resnet110' in classifiername or 'resnet164' in classifiername:
        module_list = [model.conv] + list(model.stage1) + list(model.stage2) + list(model.stage3) + [model.bn, nn.ReLU(), nn.AdaptiveAvgPool2d(output_size=1), Flatten(), model.fc]
    elif 'resnet18k' in classifiername:
        module_list = [model.conv1] + list(model.layer1) + list(model.layer2) + list(model.layer3) + list(model.layer4) + [nn.AdaptiveAvgPool2d(output_size=1), Flatten(), model.linear]
    elif 'resnet50_im' in classifiername:
        module_list = [nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])] + list(model.layer1) + list(model.layer2) + list(model.layer3) + list(model.layer4) + [model.avgpool, Flatten(), model.fc]
    else:
        raise NotImplementedError

        
        
        
    new_model = nn.Sequential(*module_list)
    return new_model

def model_stitch(model1, model2, model_name, conv_layer_num, kernel_size=1, stitch_depth=1, stitch_no_bn=False, conv_layer_num_top=None, stitch_model1_path=None, stitch_model2_path=None):
    '''
    model1 and model2 should have the same architecture
    '''

    if conv_layer_num_top is None: conv_layer_num_top = conv_layer_num

    ##### mCNN ######
    if model_name=='mCNN_k_bn_cifar10': 
        
        l1_w = model1[0].out_channels
        
        connect_units = (conv_layer_num==1)*l1_w + (conv_layer_num==2)*l1_w*2 + (conv_layer_num==3)*l1_w*4 + (conv_layer_num==4)*l1_w*8 + (conv_layer_num==5)*l1_w*16
        bottom_ind = (conv_layer_num==1)*1 + (conv_layer_num==2)*4 + (conv_layer_num==3)*8 + (conv_layer_num==4)*12 + (conv_layer_num==5)*16
        top_ind = bottom_ind

        if kernel_size==1:
            connection_layer = [nn.BatchNorm2d(connect_units),
                               nn.Conv2d(connect_units, connect_units, kernel_size=(1, 1)),
                               nn.BatchNorm2d(connect_units)]
        elif kernel_size==3:
            connection_layer = [nn.BatchNorm2d(connect_units),
                                nn.Conv2d(connect_units, connect_units, kernel_size=(3, 3), padding=1),                           
                                nn.BatchNorm2d(connect_units)]
        else:
            raise NotImplementedError

        model_centaur = nn.Sequential( *(list(model1[:bottom_ind]) + connection_layer + list(model2[top_ind:]) ))

        ### Freeze all accept connection
        for i in range(bottom_ind):
            for param in model_centaur[i].parameters():
                param.requires_grad = False
        for i in np.arange(bottom_ind+3, 23):
            for param in model_centaur[i].parameters():
                param.requires_grad = False


    ##### ResNet ########
    elif 'resnet' in model_name:

        if stitch_model1_path=='/n/coxfs01/ybansal/main-exp/freezetrain-r164-stitchcounter/p4f1p7fr/best_model.pt' or stitch_model1_path=='/n/coxfs01/ybansal/main-exp/freezetrain-r164-stitchcounter/btsbng5v/best_model.pt' or stitch_model1_path=='/n/coxfs01/ybansal/main-exp/freezetrain-r164-stitchcounter/b3mo1fcz/final_model.pt' or stitch_model1_path=='/n/coxfs01/ybansal/main-exp/freezetrain-r164-stitchcounter/is1d8q5b/final_model.pt':
            new_model1 = model1
        else:
            new_model1 = convert_model_to_Seq(model1, model_name, stitch_model1_path)
        
        new_model2 = convert_model_to_Seq(model2, model_name, stitch_model2_path)

        x = torch.randn(2, 3, 32, 32).cuda()
        #connect_units = new_model1[:conv_layer_num_top].cuda()(x).shape[1]
        connect_units_in = new_model1[:conv_layer_num].cuda()(x).shape[1]
        connect_units_out = new_model2[:conv_layer_num_top].cuda()(x).shape[1]

        if stitch_depth==1:
            if stitch_no_bn:
                connection_layer = [nn.Conv2d(connect_units_in, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2))]
            else:
                connection_layer = [nn.BatchNorm2d(connect_units_in),
                                nn.Conv2d(connect_units_in, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
                                nn.BatchNorm2d(connect_units_out)]

            if 'resnet18k' in model_name and conv_layer_num==11:
                connection_layer = [nn.BatchNorm1d(connect_units_in),
                                    nn.Linear(in_features = connect_units_in, out_features=connect_units_out, bias=True),
                                    nn.BatchNorm1d(connect_units_out)
                                    ]
            
        elif stitch_depth==2:

            connection_layer = [nn.Conv2d(connect_units_in, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
                    nn.BatchNorm2d(connect_units_in),
                    nn.ReLU(),
                    nn.Conv2d(connect_units_out, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
                    nn.BatchNorm2d(connect_units_out)]

        else:
            raise NotImplementedError

        model_centaur = nn.Sequential( *(list(new_model1[:conv_layer_num]) + connection_layer + list(new_model2[conv_layer_num_top:])) )

        for i in range(conv_layer_num):
            for param in model_centaur[i].parameters():
                param.requires_grad = False
        for i in np.arange(conv_layer_num + len(connection_layer), len(model_centaur)):
            for param in model_centaur[i].parameters():
                param.requires_grad = False

        
    else:
        raise NotImplementedError

    return model_centaur


def get_stitched_classifier(settings):

    if settings.dlaas:
        sys.path.insert(0, 'source/models')
    else:
        sys.path.insert(0, '../source/models')

    from basicmodels import create_fc, create_cnn, Flatten
    from resnet_pytorch_image_classification import resnet_pic
    from wrn import wideresnet_pic, wideresnet_mine
    from resnet import resnet18k_cifar
    from torchvision.models import resnet50


    if settings.classifiername=='vit_clip':
        stitched, clip_latent_model = stitch_vit_clip(settings.numout)
        return stitched, clip_latent_model

    if 'ImageNet' in settings.dataname:
        model2 = resnet50(pretrained=True)#torch.load(settings.stitch_model2_path, map_location='cpu')
    elif settings.classifiername=='resnet18k_cifar' and settings.stitch_model2_path=='random_init' and settings.stitch_model1_path=='random_init':
        k = settings.conv_channels
        num_classes = 10
        model2 = resnet18k_cifar(k, num_classes)
    else:
        model2 = torch.load(settings.stitch_model2_path, map_location='cpu')    

    if settings.stitch_model1_path=='random_init':
        model1 = copy.deepcopy(model2)
        for layer in model1.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    elif settings.stitch_model1_path=='pt_resnet50':
        model1 = resnet50(pretrained=True)
    elif 'vissl' in settings.stitch_model1_path:
        model1 = resnet50()
        filepath = settings.stitch_model1_path
        checkpoint = torch.load(filepath)
        try:
            model1.load_state_dict(torch.load(filepath))
        except Exception as e:
            print('Error in loading model 1' + str(e))
            pass
    elif settings.dataname=='ImageNet':
        model1 = resnet50() #torch.load(settings.stitch_model1_path, map_location='cpu')
        #filepath = os.path.join(settings.stitch_model1_path, settings.checkpoint_format.format(epoch=73))
        filepath = os.path.join(settings.stitch_model1_path, 'resnet50-1x.pth')
        checkpoint = torch.load(filepath)
        #model1.load_state_dict(checkpoint['model'])
        model1.load_state_dict(checkpoint['state_dict'])
    else:
        model1 = torch.load(settings.stitch_model1_path, map_location='cpu')

    

    model = model_stitch(model1, model2, settings.classifiername, settings.stitch_layer_num, settings.stitch_kernel_size, settings.stitch_depth, settings.stitch_no_bn, settings.stitch_layer_num_top, settings.stitch_model1_path, settings.stitch_model2_path)

    return model
