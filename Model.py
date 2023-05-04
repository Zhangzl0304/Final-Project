import torch
import torch.nn as nn
from math import sqrt

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size=11, padding=4)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size=9, padding=5)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        y = self.ReLU(self.conv1(x))
        y = self.conv2(y)
        return self.ReLU(x + y)

class Self_Attention(nn.Module):
   def __init__(self, input_size):
     super(Self_Attention, self).__init__()
     self.query = nn.Linear(input_size, input_size, bias=False)
     self.key = nn.Linear(input_size, input_size, bias=False)
     self.value = nn.Linear(input_size, input_size, bias=False)
     self.softmax = nn.Softmax(dim=2)
     self.sigmoid = nn.Sigmoid()
     self.norm_fact = 1 / sqrt(input_size)

   def forward(self, inputs):
      Q = self.query(inputs)
      K = self.key(inputs)
      V = self.value(inputs)
      K = K.permute(0, 2, 1)

      attn = torch.bmm(Q, K)*self.norm_fact

      attn = self.softmax(attn)
      output = torch.bmm(attn, V)
      return output







class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=6,
                      out_channels=32,
                      kernel_size=10,
                      dilation=3,
                      # padding='same'
                      ),
            nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 10, dilation=3,
                                             # padding='same'
                                             ),
                                   nn.LeakyReLU(),
                                   # nn.MaxPool1d(3)
                                   )
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 10, dilation=3,
                                             # padding='same'
                                             ),
                                   nn.LeakyReLU(),
                                   # nn.MaxPool1d(3)
                                   )
        self.Tconv1 = nn.Sequential(nn.ConvTranspose1d(128, 64, 10, dilation=3,
                                              # padding='same'
                                              ),
                                   nn.LeakyReLU(),
                                   )
        self.Tconv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, 10, dilation=3,
                                              # padding='same'
                                              ),
                                    nn.LeakyReLU(),
                                    )
        self.Tconv3 = nn.Sequential(nn.ConvTranspose1d(32, 6, 10, dilation=3,
                                              # padding='same'
                                              ),
                                    nn.LeakyReLU(),
                                    )
        self.Conv1 = nn.Sequential(
                      nn.Conv1d(in_channels=6,
                      out_channels=32,
                      kernel_size=8,
                      dilation=2
                              ),
                      nn.LeakyReLU(),
                      nn.MaxPool1d(kernel_size=3))
        self.Conv2 = nn.Sequential(
                      nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=8,
                      dilation=2
                              ),
                      nn.LeakyReLU(),
                      nn.MaxPool1d(kernel_size=2))
        self.Conv3 = nn.Sequential(
                      nn.Conv1d(in_channels=64,
                      out_channels=96,
                      kernel_size=8,
                      dilation=2
                              ),
                      nn.LeakyReLU(),
                      nn.MaxPool1d(kernel_size=2)
        )
        # self.space_attn = Self_Attention(250)
        # self.channel_attn = Self_Attention(6)
        # self.Res = ResidualBlock(6)
        self.norm = nn.BatchNorm1d(250)
        self.out = nn.Sequential(
            nn.Linear(864, 512),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.s_avgpool = nn.AvgPool1d(6)
        self.c_avgpool = nn.AvgPool1d(250)
    def forward(self, x):
        id = x
        x = self.norm(x)
        # id = self.norm(id)
        # x_s = x
        # x_s = self.s_avgpool(x_s)
        # x_s = x_s.permute(0, 2, 1)
        # x_s = self.space_attn(x_s)
        # x_s = x_s.permute(0, 2, 1)
        # # print(x_s.size())
        # x_c = x
        # x_c = x_c.permute(0, 2, 1)
        # x_c = self.c_avgpool(x_c)
        # x_c = x_c.permute(0, 2, 1)
        # x_c = self.channel_attn(x_c)
        # # print(x_c.size())
        # # print(x.size())
        # x = x_c*x*x_s
        x = x.permute(0, 2, 1)
        # x = self.Res(x)
        # x = self.Res(x)
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.Tconv1(x)
        # print(x.size())
        x = self.Tconv2(x)
        # print(x.size())
        x = self.Tconv3(x)
        # print(x.size())
        # id = id.permute(0, 2, 1)
        # print(x.size(), id.size())
        # x = torch.cat((x, id), dim=1)
        # x = x + id
        x = self.Conv1(x)
        print(x.size())
        x = self.Conv2(x)
        print(x.size())
        x = self.Conv3(x)
        print(x.size())
        x = x.contiguous().view(x.size(0), -1)
        output = self.out(x)
        return output


