import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Set feature extractor to inference mode
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()
        for param in self.vgg19_54.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.vgg19_54(img)


class ResidualDenseBlock(nn.Module):
    """
    Network structure of Residual Dense Block
    """

    def __init__(self, filters, res_scale=0.2):
        """
        :param filters: Number of filters in the convolution layer
        :param res_scale: Residual scaling
        """
        super(ResidualDenseBlock, self).__init__()
        self.res_scale = res_scale

        # Network structure of the block
        def block(in_feats, non_tail=True):
            layers = [nn.Conv2d(in_feats, filters, 3, 1, 1, bias=True)]
            if non_tail:
                layers += [nn.LeakyReLU(0.2, True)]
            return nn.Sequential(*layers)

        # Network structure of dense block
        self.b1 = block(in_feats=1 * filters)
        self.b2 = block(in_feats=2 * filters)
        self.b3 = block(in_feats=3 * filters)
        self.b4 = block(in_feats=4 * filters)
        self.b5 = block(in_feats=5 * filters, non_tail=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        # Residual learning
        input_info = x
        for block in self.blocks:
            out = block(input_info)
            input_info = torch.cat([input_info, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    """
    Network structure of Residual-in-Residual Dense Block
    """

    def __init__(self, filters, res_scale=0.2):
        """
        :param filters: Number of filters in the convolution layer
        :param res_scale: Residual scaling
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.residual_dense_blocks = nn.Sequential(
            ResidualDenseBlock(filters), ResidualDenseBlock(filters)
        )

    def forward(self, x):
        return self.residual_dense_blocks(x).mul(self.res_scale) + x


class Generator(nn.Module):
    """
    Generator Network Structure of SSRGAN
    """

    def __init__(self, channels, scale, n_resblocks=8, filters=64, n_upsample=1):
        """
        :param channels: Number of channels of input image
        :param scale: Downscaling scale size
        :param n_resblocks:  Number of residual-in-residual dense block in the network
        :param filters: Number of filters in the convolution layer
        :param n_upsample: Number of upsampling layers
        """
        super(Generator, self).__init__()

        # Define the head structure of the network
        # Extract the shallow features of the image
        self.head = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Residual-in-residual dense block
        # Extract the deep features of the image
        self.body = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(n_resblocks)])

        # Define the upsampling layers of the network
        upsample_layers = []
        # Facilitates expansion on multiples of existing downscaling modes
        for _ in range(n_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * scale ** 2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.PixelShuffle(upscale_factor=scale),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Define the tail structure of the network
        self.tail = nn.Sequential(
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h = self.head(x)
        x = self.body(h)
        x = torch.add(h, x)
        x = self.upsampling(x)
        x = self.tail(x)

        return x


class Discriminator(nn.Module):
    """
    Discriminator Network Structure of SSRGAN
    """

    def __init__(self, imgs_shape):
        """
        :param imgs_shape: Image information in a batch
        """
        super(Discriminator, self).__init__()

        self.imgs_shape = imgs_shape
        in_channels, in_height, in_width = self.imgs_shape
        # Round up
        patch_h, patch_w = math.ceil(in_height / 2 ** 4), math.ceil(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        # Network structure of each block in the discriminator
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            # No BN layer in the first block
            if first_block:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        # Store discriminator network
        layers = []
        in_filters = in_channels
        # Set the number of filters for the convolution layers in each block
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        # Custom sequential connection model
        self.ssrgan_discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.ssrgan_discriminator(x)
