from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock


class SpatiotemporalConv(nn.Module):
    """
    Spatiotemporal conv layer to process video stream
    """

    def __init__(self):
        super(SpatiotemporalConv, self).__init__()
        # 29x112x112 -> 29x56x56
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2), padding=(2, 3, 3))
        self.norm = nn.BatchNorm3d(64)
        # 29x56x56 -> 29x28x28
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2),
                                 padding=(0, 1, 1))

    def forward(self, x):
        """
        x: input lip video, with shape [Batch-BS, frames-FR, 1, height-H, width-W]
        y: output with shape [Batch-BS, out_channel, frames-FR, H', W']
        """
        # NxTx1xDxD => Nx1xTxDxD
        # [BS, FR, 1, H, W] => [BS, 1, FR, H, W]
        x = torch.transpose(x, 1, 2)
#	print(x.shape)
#	x = x.permute((0, 2, 1, 3, 4))
        # input: [BS, in_channel=1, Depth-FR, H, W]
        # output: [BS, out_channel=64, Depth-FR, H', W']
        x = self.conv(x)	# here, batch size cannot be too large otherwise will raise CUDNN_STATUS_NOT_SUPPORTED error, maybe a cuDnn bug

        x = self.norm(x)
        x = F.relu(x)

        x = self.pool(x)
        return x


class LipReadingNet(nn.Module):
    """
    Lip reading phoneme level networks
    """

    def __init__(self, backend_dim=256):
        super(LipReadingNet, self).__init__()
        self.conv3d = SpatiotemporalConv()
        self.resnet = resnet18(num_classes=backend_dim)

    def forward(self, x, return_embedding=False):
        """
        x: input lip video, with shape [Batch-BS, frames-FR, 1, height-H, width-W]
        y: lip embedding, with shape [Batch-BS, frames-FR, embedding_dim]
        """
        if x.dim() != 5:
            raise RuntimeError(
                "LipReadingNet accept 5D tensor as input, got {:d}".format(
                    x.dim()))
        # input: lip video, with shape [Batch-BS, frames-FR, 1, height-H, width-W]
        # output: with shape [Batch-BS, out_channel, frames-FR, H', W']
        x = self.conv3d(x)

        # [BS, out_channel, FR, H', W'] => [BS, FR, out_channel, H', W']
        x = torch.transpose(x, 1, 2)

        BS, FR, C, D1, D2 = x.shape[:5]
        x = x.reshape(BS * FR, C, D1, D2)

        # [BS*FR, C, H', W'] => [BS*FR, Z]
        x = self.resnet(x)

        if return_embedding:
            return x

        # Batch-BS, frames-FR, embedding_dim
        x = x.view(BS, FR, -1)

        return x


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=512,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)



class OxfordLipConv1DBlock(nn.Module):
    """
    depthwise pre-activation conv1d block used in OxfordLipNet
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=256,
                 kernel_size=3,
                 dilation=1):
        super(OxfordLipConv1DBlock, self).__init__()
        self.residual = (in_channels == conv_channels)
        self.bn = nn.BatchNorm1d(in_channels) if self.residual else None
        self.prelu = nn.PReLU() if self.residual else None
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        self.sconv = nn.Conv1d(in_channels, conv_channels, 1)

    def forward(self, x):
        if self.residual:
            y = self.dconv(self.bn(self.prelu(x)))
            y = self.sconv(y) + x
        else:
            y = self.dconv(x)
            y = self.sconv(y)
        return y


class OxfordLipNet(nn.Module):
    """
    Oxford like lip net to process lip embeddings
    """

    def __init__(self,
                 embedding_dim=256,
                 conv_channels=256,
                 kernel_size=3,
                 num_blocks=5):
        super(OxfordLipNet, self).__init__()
        conv1d_list = []
        for i in range(num_blocks):
            in_channels = conv_channels if i else embedding_dim
            conv1d_list.append(
                OxfordLipConv1DBlock(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size))
        self.conv1d_blocks = nn.Sequential(*conv1d_list)

    def forward(self, x):
        '''
        x: input lip embedding, with shape [Batch size (BS), frames (FR), embedding_dim (D)]
        y: output lip embedding, with shape [Batch size (BS), embedding_dim (D), frames (FR)]
        '''
        x = torch.transpose(x, 1, 2)
        x = self.conv1d_blocks(x)
        return x


def foo_spatialTemporal_conv():
    bs = 5
    frames = 80
    height, width = 112, 112
    x = torch.rand(bs, frames, 1, height, width)
    stc = SpatiotemporalConv()
    y = stc(x)
    print(y.shape)  # [BS, out_channel, frames, 28, 28]


def foo_lipreadingNet():
    bs = 5
    frames = 80
    height, width = 112, 112
    x = torch.rand(bs, frames, 1, height, width)
    nnet = LipReadingNet()
    y = nnet(x)
    print(y.shape)

def foo_videoNet():
    bs = 5
    frames = 80
    height, width = 112, 112
    x = torch.rand(bs, frames, 1, height, width)

    lipReadNet = LipReadingNet()
    lipNet = OxfordLipNet()

    y = lipNet(lipReadNet(x))
    print(y.shape)


if __name__ == '__main__':
    # foo_spatialTemporal_conv()
    # foo_lipreadingNet()
    foo_videoNet()
