import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True)
            )
        #Encoder
        self.e1 = CBR(3, 32)
        self.e2 = CBR(32, 64)
        self.e3 = CBR(64, 128)

        self.pool = nn.MaxPool2d(2, 2)

        #Decoder
        self.d3 = CBR(128, 64)
        self.d2 = CBR(64, 32)
        self.d1 = nn.Conv2d(32, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        e1 = self.e1(x)
        p1 = self.pool(e1)

        e2 = self.e2(p1)
        p2 = self.pool(e2)

        e3 = self.e3(p2)

        d3 = self.up(e3)
        d3 = self.d3(d3)

        d2 = self.up(d3)
        d2 = self.d2(d2)

        out = torch.sigmoid(self.d1(d2))
        return out
