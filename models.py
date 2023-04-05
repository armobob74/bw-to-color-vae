import torch
import torch.nn as nn
import torch.optim as optim


class BaseEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32*32, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, 128*32*32)
        x = nn.functional.relu(self.fc1(x))
        mean = self.fc21(x)
        logvar = self.fc22(x)
        return mean, logvar


class BaseDecoder(nn.Module):
    def __init__(self, output_channels, latent_dim):
        super(BaseDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128*32*32)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.relu(self.fc2(z))
        z = z.view(-1, 128, 32, 32)
        z = nn.functional.relu(self.conv1(z))
        z = nn.functional.relu(self.conv2(z))
        x = torch.sigmoid(self.conv3(z))
        return x


class BaseVAE(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(BaseVAE, self).__init__()
        self.encoder = BaseEncoder(input_channels, latent_dim)
        self.decoder = BaseDecoder(output_channels, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps * std + mean
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu


class ABVAE(BaseVAE):
    def __init__(self, latent_dim):
        super(ABVAE, self).__init__(input_channels=2, latent_dim=latent_dim, output_channels=2)


class LVAE(BaseVAE):
    def __init__(self, latent_dim):
        super(LVAE, self).__init__(input_channels=1, latent_dim=latent_dim, output_channels=1)

    