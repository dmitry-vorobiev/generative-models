from torch import nn

from .layers import EqualLeakyReLU, EqualLinear, ModulatedConv2d, RandomGaussianNoise


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, up=False):
        super(ToRGB, self).__init__()
        if up:
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upscale = None

        self.style = EqualLinear(style_dim, in_channels, bias=True)
        nn.init.ones_(self.style.bias)

        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=1,
                                    stride=1, padding=0, demodulate=False)

    def forward(self, x, w, x0=None):
        y = self.style(w)
        x = self.conv(x, y)
        if x0 is not None:
            if self.upscale:
                x0 = self.upscale(x0)
            x = x + x0
        return x


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, up=False):
        super(Layer, self).__init__()
        if up:
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upscale = None

        self.style = EqualLinear(style_dim, in_channels, bias=True)
        nn.init.ones_(self.style.bias)

        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1)
        self.add_noise = RandomGaussianNoise()
        self.act_fn = EqualLeakyReLU(inplace=True)

    def forward(self, x, w):
        if self.upscale:
            x = self.upscale(x)
        y = self.style(w)
        x = self.conv(x, y)
        x = self.act_fn(self.add_noise(x))
        return x
