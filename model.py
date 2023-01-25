from imports import *

"""
Building block of the UNet Model
"""
class Block(nn.Module):
    def __init__(self, in_c, out_c, time_emb=32):
        super().__init__()
        # define layers
        self.time_mlp =  nn.Linear(time_emb, out_c)
        self.conv1  = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_c)
        self.bnorm2 = nn.BatchNorm2d(out_c)
        self.relu   = nn.ReLU()

    def forward(self, x):
        # first convolution
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        # second convolution
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        return x

"""
Building block of the encoder network
"""
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # define layers
        self.conv = Block(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        # keep track of the skip connection
        skip = x.clone()
        x = self.pool(x)
        return x, skip

"""
Building block of the decoder network
"""
class DecoderBlock(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.up    = nn.ConvTranspose2d(num_in, num_out, kernel_size=2, stride=2, padding=0)
        self.block = Block(num_out + num_out, num_out)

    def forward(self, x, skip):
        x = self.up(x)
        # concatenate the skip connection after the upsample
        x = torch.cat([x, skip], axis=1)
        x = self.block(x)
        return x

"""
The final UNet network
"""
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # the down/up channels used
        down_channels = (1, 64, 128, 256, 512)
        up_channels =   (1024, 512, 256, 128, 64)

        self.downs = nn.ModuleList([EncoderBlock(down_channels[i], down_channels[i+1]) for i in range(len(down_channels)-1)])
        self.ups   = nn.ModuleList([DecoderBlock(up_channels[i], up_channels[i+1]) for i in range(len(up_channels)-1)])
        
        self.bottle_neck = Block(down_channels[-1], up_channels[0])

        self.out = nn.Conv2d(up_channels[-1], 1, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        residuals = []
        for down in self.downs:
            x, skip = down(x)
            residuals.append(skip)

        # Bottle neck
        x = self.bottle_neck(x)

        # Decoder
        for up in self.ups:
            x = up(x,residuals.pop())

        return self.out(x)