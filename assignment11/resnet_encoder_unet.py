import torchvision
import torch.nn as nn
import torch

# resnet = torchvision.models.resnet.resnet50(pretrained=True)

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    )
    return model


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels)
    )
    return model

###########################################################################
# Code overlaps with previous assignments : Implement the "bottle neck building block" part.
# Hint : Think about difference between downsample True and False. How we make the difference by code?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.layer = nn.Sequential(
                ##########################################
                ############## fill in here
                # Hint : use these functions (conv1x1, conv3x3)
                #########################################
                conv1x1(in_channels, middle_channels, stride=2, padding=0),  # in -> middle(다운샘플링 有 -> S=2)
                conv3x3(middle_channels, middle_channels, stride=1, padding=1),  # middle -> middle(채널 수 유지 -> S=1)
                conv1x1(middle_channels, out_channels, stride=1, padding=0)  # middle -> out(채널 유지 -> S=1)
            )
            self.downsize = conv1x1(in_channels, out_channels, 2, 0)

        else:
            self.layer = nn.Sequential(
                ##########################################
                ############# fill in here
                #########################################
                conv1x1(in_channels, middle_channels, stride=1, padding=0),
                conv3x3(middle_channels, middle_channels, stride=1, padding=1),
                conv1x1(middle_channels, out_channels, stride=1, padding=0)
            )
            self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            out = self.layer(x)
            x = self.downsize(x)
            return self.activation(out + x)     # This part is slightly different from previous assignments of 'OSP-Lec14-CNN architecture-practice-v2.pdf'
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return self.activation(out + x)     # This part is slightly different from previous assignments of 'OSP-Lec14-CNN architecture-practice-v2.pdf'

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3: kernel size
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),  # When inplace = TRUE, ReLU modifies input activations, without allocating additional outputs. This often decrease the memory usage, but may sometimes cause some errors.
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNetWithResnet50Encoder(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            # nn.Conv2d(#blank#, #blank#, #blank#, #blank#, #blank# ), # Code overlaps with previous assignments
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)#,
        )
        self.pool = nn.MaxPool2d(3, 2, 1, return_indices=True)

        self.layer2 = nn.Sequential(
            # ResidualBlock(#blank#, #blank#, #blank#),
            # ResidualBlock(#blank#, #blank#, #blank#),
            # ResidualBlock(#blank#, #blank#, #blank#, downsample=True) # Code overlaps with previous assignments
            ResidualBlock(64, 64, 256, False),
            ResidualBlock(256, 64, 256, False),
            ResidualBlock(256, 64, 256, True)
        )
        self.layer3 = nn.Sequential(
            # ResidualBlock(#blank#, #blank#, #blank#),
            # ResidualBlock(#blank#, #blank#, #blank#),
            # ResidualBlock(#blank#, #blank#, #blank#),
            # ResidualBlock(#blank#, #blank#, #blank#, downsample=False) # Code overlaps with previous assignments
            ResidualBlock(256, 128, 512, False),
            ResidualBlock(512, 128, 512, False),
            ResidualBlock(512, 128, 512, False),
            ResidualBlock(512, 128, 512, False)
        )
        self.bridge = conv(512, 512)
        self.UnetConv1 = conv(512, 256)
        self.UpConv1 = nn.Conv2d(512, 256, 3, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(256, 256, 3, 2, 1)
        self.upconv2_2 = nn.Conv2d(256, 64, 3, padding=1)

        self.unpool = nn.MaxUnpool2d(3, 2, 1)
        self.UnetConv2_1 = nn.ConvTranspose2d(64, 64, 3, 2, 1)
        self.UnetConv2_2 = nn.ConvTranspose2d(128, 128, 3, 2, 1)
        self.UnetConv2_3 = nn.Conv2d(128, 64, 3, padding=1)

        self.UnetConv3 = nn.Conv2d(64, self.n_classes, kernel_size=1, stride=1)

    ###########################################################################
    # Question 2 : Implement the forward function of Resnet_encoder_UNet.
    # Understand ResNet, UNet architecture and fill in the blanks below. (20 points)
    def forward(self, x, with_output_feature_map=False): #256

        out1 = self.layer1(x)
        out1, indices = self.pool(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.bridge(out3) # bridge
        x = self.UpConv1(x)
        # x = #######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        '''
        "Every step in the expansive path consists of an umsampling of the feature map followed by a 2x2 conv
        that halves the # of feature channels, a concat. with the correspondingly cropped feature map from the
        contracting path, and 3x3 conv, each followed by a ReLU"
        '''
        x = torch.cat([x, out2], dim=1)     # out2와 concat
        x = self.UnetConv1(x)
        x = self.upconv2_1(x, output_size=torch.Size([x.size(0),256,64,64]))
        x = self.upconv2_2(x)
        # x = #######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = torch.cat([x, out1], dim=1) # out1과 concat
        x = self.UnetConv2_2(x, output_size=torch.Size([x.size(0), 128, 128, 128]))
        x = self.UnetConv2_2(x, output_size=torch.Size([x.size(0), 128, 256, 256]))
        x = self.UnetConv2_3(x)
        x = self.UnetConv3(x)
        return x

