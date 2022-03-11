import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.H, self.W, self.C = img_shape
        self.condition_resize = nn.Sequential(
            nn.Linear(24, self.H*self.W*1),
            nn.ReLU()
        )

        channels = [4,64,128,256,512]
        for i in range(1, len(channels)):
            setattr(self, 'conv'+str(i), nn.Sequential(
                nn.Conv2d(channels[i-1], channels[i], kernel_size=(4, 4), stride=(2, 2),padding=(1, 1)),
                nn.BatchNorm2d(channels[i]),
                nn.LeakyReLU()
            ))
        self.conv5 = nn.Conv2d(512, 1, kernel_size=(4, 4))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        #   make condition be (N, 1, 64, 64)
        c = self.condition_resize(c.float()).view(-1, 1, self.H, self.W)
        #   (N, 4, 64, 64)
        output = torch.cat((x, c), dim=1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        #   (N, 1, 1, 1)
        output = self.conv5(output)
        output = self.sigmoid(output)
        #   true / false
        output = output.view(-1)
    
        return output

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.condition_resize = nn.Sequential(
            nn.Linear(24, c_dim),
            nn.ReLU()
        )

        channels = [z_dim+c_dim,512,256,128,64]
        paddings=[(0,0),(1,1),(1,1),(1,1)]
        for i in range(1,len(channels)):
            setattr(self,'conv'+str(i),nn.Sequential(
                nn.ConvTranspose2d(channels[i-1], channels[i], kernel_size=(4, 4), stride=(2, 2), padding=paddings[i-1]),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU()
            ))
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tanh=nn.Tanh()

    def forward(self, z, c):
        z = z.view(-1, self.z_dim, 1, 1)
        c = self.condition_resize(c).view(-1, self.c_dim, 1, 1)
        output = torch.cat((z, c), dim=1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.tanh(output)

        return output
