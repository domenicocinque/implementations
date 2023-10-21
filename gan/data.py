from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import tensorflow as tf


def generate_circle_data(
    key: random.PRNGKey, n_samples: int, r: int = 1, noise: float = 0.05
):
    subkey1, subkey2, subkey3 = random.split(key, 3)
    theta = jax.random.uniform(
        key=subkey1, shape=(n_samples,), minval=0, maxval=2 * jnp.pi
    )
    x_noise = jax.random.normal(key=subkey2, shape=(n_samples,)) * noise
    x = r * jnp.cos(theta) + x_noise
    y_noise = jax.random.normal(key=subkey3, shape=(n_samples,)) * noise
    y = r * jnp.sin(theta) + y_noise
    return jnp.stack([x, y], axis=1)


def get_dataloader(dataset: Tuple[np.ndarray, ...], batch_size: int):
    return (
        tf.data.Dataset.from_tensor_slices(dataset)
        .shuffle(2000)
        .batch(batch_size, drop_remainder=True)
        .as_numpy_iterator()
    )


if __name__ == "__main__":
    train_examples = generate_circle_data(key=random.PRNGKey(0), n_samples=1000)

    for epoch in range(1):
        train_dataset = get_dataloader(train_examples, batch_size=32)  
        for step, batch in enumerate(train_dataset):
            if step == 0:
                print(type(batch))
                print(batch.shape)
