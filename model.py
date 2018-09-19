import torch
import torch.nn as nn
import conf
import helpers

# encodes the visual state into a lower dimension abstract representation
class Encoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.Conv2d(conf.FRAMES_PER_STATE, 32, kernel_size=(8,8), stride=(4,4)),
            nn.ELU(),

            nn.Conv2d(32, 40, kernel_size=(4,4), stride=(2,2)),
            nn.ELU(),

            nn.Conv2d(40, 50, kernel_size=(3,3), stride=(1,1)),
            nn.ELU(),
        )

        self.flat_size = helpers.flat_size_after_conv(self.conv, conf.IMAGE_WIDTH,conf.IMAGE_HEIGHT)

        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ELU(),
            nn.Linear(512, conf.HIDDEN_SIZE)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,self.flat_size)
        x = self.fc(x)
        return x

class InverseModel(nn.Module):
    def __init__(self, encoder):
        nn.Module.__init__(self)
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Linear(conf.HIDDEN_SIZE*2, 512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256, conf.OUTPUT_NUM)
        )

    def forward(self, x1,x2):
        enc1 = self.encoder(x1)
        enc2 = self.encoder(x2)

        enc_cat = torch.cat((enc1,enc2), dim=1) # cat along channel
        pred_action = self.fc(enc_cat)

        return pred_action
