from functools import partial

import optax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from omegaconf import OmegaConf

from data import get_dataloader, generate_circle_data
from model import generator_model, discriminator_model
from utils import save_gan_training_gif


@jit
def generator_step(gen_state, disc_state, latent_key):
    def loss_fn(params):
        fake_data = gen_state.apply_fn(params, latent_key)
        fake_logits = disc_state.apply_fn(disc_state.params, fake_data)

        # In the non-saturating loss, we want to maximize the probability that
        # the discriminator classifies the fake data as real
        loss = -jnp.mean(jnp.log(nn.sigmoid(fake_logits)))
        return loss

    loss, grads = value_and_grad(loss_fn)(gen_state.params)
    gen_state = gen_state.apply_gradients(grads=grads)
    return gen_state, loss


@jit
def discriminator_step(gen_state, disc_state, batch, latent_key):
    fake_data = gen_state.apply_fn(gen_state.params, latent_key)

    def loss_fn(params):
        fake_logits = disc_state.apply_fn(params, fake_data)
        real_logits = disc_state.apply_fn(params, batch)

        fake_loss = optax.sigmoid_binary_cross_entropy(
            fake_logits, jnp.zeros_like(fake_logits)
        )
        real_loss = optax.sigmoid_binary_cross_entropy(
            real_logits, jnp.ones_like(real_logits)
        )
        loss = jnp.mean(fake_loss + real_loss)
        return loss

    loss, grads = value_and_grad(loss_fn)(disc_state.params)
    disc_state = disc_state.apply_gradients(grads=grads)
    return disc_state, loss


def train_step(gen_state, disc_state, batch, rng, cfg):
    rng, gen_key, disc_key = random.split(rng, 3)

    for _ in range(cfg.disc_steps):
        # We need to re-initialize the key for the latent space,
        # otherwise the generator will always generate the same data
        rng, gen_key = random.split(rng)
        disc_state, disc_loss = discriminator_step(
            gen_state, disc_state, batch, disc_key
        )
    gen_state, gen_loss = generator_step(gen_state, disc_state, gen_key)

    return gen_state, disc_state, gen_loss, disc_loss, rng


def fit(cfg):
    # Init random key
    rng = random.key(0)
    rng, key = random.split(rng)

    # Init data
    training_data = generate_circle_data(key=key, n_samples=cfg.num_samples)

    # Init model
    # We need to split the key for the generator and discriminator
    rng, gen_key, disc_key = random.split(rng, 3)

    # We use partial to pass the hidden_channels and batch_size to the model
    generator = partial(
        generator_model, hidden_channels=cfg.hidden_dims, batch_size=cfg.batch_size
    )
    discriminator = partial(discriminator_model, hidden_channels=cfg.hidden_dims)

    # Init parameters by passing a dummy input input
    generator_params = generator().init(rngs=gen_key, z_rng=gen_key)
    discriminator_params = discriminator().init(
        disc_key, jnp.ones((cfg.batch_size, 2), dtype=jnp.float32)
    )

    # Instantiate training states
    gen_state = train_state.TrainState.create(
        apply_fn=generator().apply,
        params=generator_params,
        tx=optax.adam(learning_rate=cfg.gen_lr),
    )

    disc_state = train_state.TrainState.create(
        apply_fn=discriminator().apply,
        params=discriminator_params,
        tx=optax.adam(learning_rate=cfg.disc_lr),
    )

    # For plotting
    predictions_list = []

    # Training loop
    for epoch in range(cfg.num_epochs):
        train_loader = get_dataloader(training_data, batch_size=cfg.batch_size)
        for batch in train_loader:
            gen_state, disc_state, gen_loss, disc_loss, rng = train_step(
                gen_state, disc_state, batch, key, cfg
            )

        print(
            f"Epoch {epoch} | "
            f"Generator loss: {gen_loss:.2f} | "
            f"Discriminator loss: {disc_loss:.2f}"
        )

        rng, key = random.split(rng)

        # Generate some data for visualization
        fake_data = generator_model(cfg.hidden_dims, cfg.batch_size).apply(
            gen_state.params, key
        )
        predictions_list.append(fake_data)

        rng, key = random.split(rng)

    save_gan_training_gif(training_data, predictions_list)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    fit(cfg)
