import flax.linen as nn
from jax import random


class Generator(nn.Module):
    hidden_channels: int
    batch_size: int

    @nn.compact
    def __call__(self, z_rng):
        z = random.normal(z_rng, (self.batch_size, 2))
        z = nn.Dense(self.hidden_channels)(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(self.hidden_channels)(z)
        z = nn.leaky_relu(z)
        x = nn.Dense(2)(z)
        return x


class Discriminator(nn.Module):
    hidden_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_channels)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_channels)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(2)(x)
        return x


def generator_model(hidden_channels, batch_size):
    return Generator(hidden_channels=hidden_channels, batch_size=batch_size)


def discriminator_model(hidden_channels):
    return Discriminator(hidden_channels=hidden_channels)
