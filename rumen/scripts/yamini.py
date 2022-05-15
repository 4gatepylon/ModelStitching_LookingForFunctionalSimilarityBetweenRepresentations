class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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

def model_stitch(model1, model2, model_name, conv_layer_num, kernel_size=1, stitch_depth=1, stitch_no_bn=False, conv_layer_num_top=None, stitch_model1_path=None, stitch_model2_path=None):
    '''
    model1 and model2 should have the same architecture
    '''

    if conv_layer_num_top is None:
        conv_layer_num_top = conv_layer_num

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