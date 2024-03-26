import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Session9Net(nn.Module):
    
    def __init__(self):
        super(Session9Net, self).__init__()
        
        # Convolutional Block-1
        #Prepare data for conv block 1
        in_channels = 3
        out_channels_list = [32, 32, 32]
        kernel_size_list = [3, 3, 3]
        stride_list = [1, 1, 2]
        padding_list = [1, 2, 1]
        dilation_list = [0,2,2]
        conv_type = ['standard','dilated','dilated']
        self.conv_block1 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)

        # Convolutional Block-2
        #Prepare data for conv block 2
        in_channels = 32
        out_channels_list = [38, 38, 38]
        kernel_size_list = [3, 3, 3]
        stride_list = [1, 1, 2]
        padding_list = [1, 2, 1]
        dilation_list = [0,2,2]
        conv_type = ['standard','dilated','dilated']
        self.conv_block2 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)

        # Convolutional Block-3
        #Prepare data for conv block 3
        in_channels = 38
        out_channels_list = [40, 40, 40]
        kernel_size_list = [3, 3, 3]
        stride_list = [1, 1, 2]
        padding_list = [1, 2, 1]
        dilation_list = [0,2,2]
        conv_type = ['depthwise','dilated','dilated']
        self.conv_block3 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)

        # Convolutional Block-4
        #Prepare data for conv block 4
        in_channels = 40
        out_channels_list = [64, 64, 64]
        kernel_size_list = [3, 3, 3]
        stride_list = [1, 1, 1]
        padding_list = [1, 2, 2]
        dilation_list = [0,2,2]
        conv_type = ['standard','dilated','dilated']
        self.conv_block4 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)

        # Global Average Pooling
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        # Convolutional Block- Output
        #Prepare data for conv block Output
        in_channels = 64
        out_channels_list = [10]
        kernel_size_list = [1]
        stride_list = [1]
        padding_list = [0]
        dilation_list = [0]
        conv_type = ['standard']
        self.conv_block_output = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=True)
    

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.gap(x)
        x = self.conv_block_output(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
        
    def get_conv_block(self, conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False):
        assert len(out_channels_list) == len(kernel_size_list) == len(stride_list) == len(padding_list), "Lengths of lists should match"
        layers = []
            
        for i in range(len(out_channels_list)):
            if i == 0:
                if conv_type[i] == "standard":
                    _conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_list[i], stride=stride_list[i], kernel_size=kernel_size_list[i], bias=False, padding=padding_list[i])
                elif conv_type[i] == "depthwise":
                    _conv_layer = self.depthwise_conv(in_channels=in_channels, out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i])
                elif conv_type[i] == "dilated":
                    _conv_layer = self.dilated_conv(in_channels=in_channels, out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i], dilation=dilation_list[i])
        
                layers.append(_conv_layer)
            else:
                if conv_type[i] == "standard":
                    _conv_layer = nn.Conv2d(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i], stride=stride_list[i], kernel_size=kernel_size_list[i], bias=False, padding=padding_list[i])
                elif conv_type[i] == "depthwise":
                    _conv_layer = self.depthwise_conv(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i])
                elif conv_type[i] == "dilated":
                    _conv_layer = self.dilated_conv(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i], dilation=dilation_list[i])
        
                layers.append(_conv_layer)
                    

            if not last_layer:
                _norm_layer = self.get_normalization_layer(normalization,out_channels_list[i],number_of_groups)
                layers.append(_norm_layer)
                layers.append(activation_fn)
                
        conv_layers = nn.Sequential(*layers)
        return conv_layers
        

    @staticmethod
    def get_normalization_layer(normalization,out_channels,number_of_groups = None):
        if normalization == "layer":
            _norm_layer = nn.GroupNorm(1, out_channels)
        elif normalization == "group":
            if not number_of_groups:
                raise ValueError("Value of group is not defined")
            _norm_layer = nn.GroupNorm(number_of_groups, out_channels)
        else:
            _norm_layer = nn.BatchNorm2d(out_channels)
        
        return _norm_layer
    

        
    @staticmethod
    def depthwise_conv(in_channels, out_channels, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels, kernel_size=3, bias=False, padding=padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1, bias=False, padding=0)
        )

    @staticmethod
    def dilated_conv(in_channels, out_channels, stride=1, padding=0, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, bias=False,
                      padding=padding, dilation=dilation)
        )

    
def get_summary(model, input_size) :       
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    return summary(network, input_size=input_size)

    
#unit test


model = Session9Net()
input_tensor = torch.randn(1, 3, 224, 224)  
output_tensor = model(input_tensor)
print(output_tensor.shape)
