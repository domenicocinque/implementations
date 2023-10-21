from typing import List
from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def save_gan_training_gif(training_data: jnp.ndarray, predictions_list: List):
    fig, ax = plt.subplots()
    ax.set_title("GAN training on a circle dataset")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.grid(True, alpha=0.3)

    # Plot the real data
    ax.scatter(
        training_data[:, 0],
        training_data[:, 1],
        color="blue",
        alpha=0.3,
        label="Training data",
    )

    # Plot the predictions
    scat = ax.scatter(
        predictions_list[0][:, 0],
        predictions_list[0][:, 1],
        color="red",
        alpha=0.2,
        label="Generated data",
    )

    def update(i):
        scat.set_offsets(predictions_list[i])
        return (scat,)

    ax.legend(loc="lower right")
    anim = animation.FuncAnimation(
        fig, update, frames=len(predictions_list), interval=100
    )
    anim.save("animation.gif", dpi=80)
