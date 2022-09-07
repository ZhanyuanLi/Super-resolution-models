import torch.nn as nn


class Block(nn.Module):
    """
    Structure of the Single Residual Block
    """

    def __init__(self, filters, kernel_size, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        # Residual scaling
        self.res_scale = res_scale
        # Body structure
        self.body = nn.Sequential(
            nn.Conv2d(filters, filters * 6, 1, padding=0),
            act,
            nn.Conv2d(filters * 6, int(filters * 0.8), 1, padding=0),
            nn.Conv2d(int(filters * 0.8), filters, kernel_size, padding=kernel_size // 2)

        )

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class ModelFADSR(nn.Module):
    """
    FADSR Network Structure: Feature Extraction Layer + Image Reconstruction Layer
    """

    def __init__(self, channels=3, scale=5, n_resblocks=8, filters=32, kernel_size=3, res_scale=1):
        """
        :param channels: Number of channels of input image
        :param scale: Downscaling scale size
        :param n_resblocks: Number of residual blocks in the network
        :param filters: Number of filters in the convolution layer
        :param kernel_size: Convolution kernel size
        :param res_scale: Residual scaling
        """
        super(ModelFADSR, self).__init__()
        act_fun = nn.LeakyReLU(0.2, True)

        # ----------
        #  Feature Extraction Layer
        # ----------
        # Define the head structure of the network
        self.head = nn.Sequential(nn.Conv2d(channels, filters, 3, padding=1))

        # Define the body structure of the network
        body = []
        for _ in range(n_resblocks):
            body.append(Block(filters, kernel_size, act=act_fun, res_scale=res_scale))
        self.body = nn.Sequential(*body)

        # Define the tail structure of the network
        feats = scale * scale * channels
        self.tail = nn.Sequential(
            nn.Conv2d(filters, feats, 3, padding=1),
            nn.PixelShuffle(scale),
        )

        # ----------
        #  Image Reconstruction Layer
        # ----------
        self.skip = nn.Sequential(
            nn.Conv2d(channels, feats, 5, padding=2),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s

        return x
