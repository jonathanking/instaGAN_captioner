import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import BCELoss

"""
Run this block of code before loading the model to define the pytorch nn module.
"""

nz = 100
ngf = 64
ndf = 64
nc = 3


def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        output = self.main(z)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(ndf * 8, ndf * 4, 2, 2, 0, bias=False)
        )

        self.post_main = nn.Sequential(
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.fc1 = nn.Linear(1024, nz)
        self.adversarial_disc = nn.Linear(nz, 1)

    def forward(self, x):
        encoded = self.main(x)
        output = self.post_main(encoded)
        output = output.view(-1, 1).squeeze(1)
        encoded_flat = encoded.view(encoded.shape[0], -1)
        return output, encoded_flat


"""
GAN class has two components - generator and discriminator
To embed an image:

disc, embed = gan.discriminator(X)

where disc is the discriminator's sigmoid output, embed contains the embedded vectors
of batch X, and gan is the gan model you loaded.
"""


class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def fit(self, X, n_epochs=1, batch_size=32, loss_function=BCELoss(), verbose=True, lr=0.0002):

        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        g_losses, d_losses = [], []

        prev_g_loss, prev_d_loss = 0, 0

        for n in range(n_epochs):

            batch_indices = np.random.permutation(X.shape[0])

            d_running_loss = 0
            g_running_loss = 0
            n_items = 0

            for i in range(0, X.shape[0], batch_size):
                idx = batch_indices[i:i + batch_size]

                x = X[idx].cuda()

                # discriminator step
                disc_optimizer.zero_grad()

                valid_y = to_cuda(torch.ones((x.shape[0],)))
                fake_y = to_cuda(torch.zeros((x.shape[0],)))

                output, encoded = gan.discriminator(x)

                loss_real = loss_function(output, valid_y)
                loss_real.backward(retain_graph=True)

                fake_z = to_cuda(torch.randn(x.shape[0], nz, 1, 1))
                fake_x = gan.generator(fake_z)

                output, encoded = gan.discriminator(fake_x)

                loss_fake = loss_function(output, fake_y)
                loss_fake.backward(retain_graph=True)

                error_d = loss_real + loss_fake
                disc_optimizer.step()

                # generator step
                gen_optimizer.zero_grad()

                output, encoded = gan.discriminator(fake_x)
                error_g = loss_function(output, valid_y)

                error_g.backward(retain_graph=True)

                gen_optimizer.step()

                d_running_loss += float(error_d) * x.shape[0]
                g_running_loss += float(error_g) * x.shape[0]
                n_items += x.shape[0]

                prev_g_loss = g_running_loss / n_items
                prev_d_loss = d_running_loss / n_items

                if verbose:
                    print('Epoch {}: {:.3f}% complete - D loss: {:.4f} - G loss: {:.4f}'.format(
                        n + 1, 100 * n_items / X.shape[0], d_running_loss / n_items, g_running_loss / n_items
                    ), end='\r')

            print()

            d_losses.append(d_running_loss / n_items)
            g_losses.append(g_running_loss / n_items)

        return d_losses, g_losses


""" Load GAN model and store into variable `gan`. Remove `map_location` to store model on CUDA device """
# gan = torch.load('DCGAN_embed_2.tch', map_location='cpu')
