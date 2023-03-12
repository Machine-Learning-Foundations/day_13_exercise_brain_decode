"""Train eeg brain decoder."""
import os
import sys
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm

sys.path.append(".")

from src.load_eeg import load_train_valid_test


def normalize(
    data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None.
            Defaults to None.
        std (float): Data standard deviation,
            re-computed if None. Defaults to None.

    Returns:
        np.array, float, float: Normalized data, mean and std.
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std, mean, std


class BrainCNN(nn.Module):
    """Your Brain-CNN model."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the forward pass."""
        # TODO choose a suitable network architecture.
        return x


@jax.jit
def forward_pass(
    variables: FrozenDict, eeg_batch: jnp.ndarray, label_batch: jnp.ndarray
):
    """Evaluate the network forward pass and compute the cost function."""
    # TODO: Run cnn.apply to apply the weights to the network.
    # TODO: Use jnp.mean and optax.softmax_cross_entropy to compute the cost.
    ce_loss = 0.0  # TODO: Find the actual cost value.
    return ce_loss


@jax.jit
def adjust_dimensions(eeg_input: jnp.ndarray) -> jnp.ndarray:
    """Adjust the eeg_input dimensions to allow CNN processing.

    Args:
        eeg_input (jnp.ndarray): The eeg input array.

    Returns:
        jnp.ndarray: An output array suitable to a CNN's liking.
    """
    return jnp.expand_dims(jnp.array(eeg_input).transpose(0, 2, 1), -3)


def get_acc(
    net: nn.Module, variables: FrozenDict, eeg_input: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    """Compute the accuracy.

    Args:
        net (nn.Module): Your cnn object.
        variables (FrozenDict): The network weights.
        eeg_input (jnp.ndarray): An array containing the eeg brain waves.
        labels (jnp.ndarray): The action annotations in a array.

    Returns:
        jnp.ndarray: The accuracy in [%].
    """
    # TODO: compute the network output using net.apply
    # TODO: find the accuracy usig jnp.argmax
    accuracy = jnp.array(0.0)  # Return the actual accuracy instead of zero.
    return accuracy


if __name__ == "__main__":
    low_cut_hz = 0  # Do not touch this.
    subject_id = 1  # Feel free to change the participant id.
    batch_size = 50  # Feel free to play with this value.
    learning_rate = 0.001  # Feel free to play with this value.
    epochs = None  # choose a suitable number of epochs.

    train_filename = os.path.join("./data", "train/{:d}.mat".format(subject_id))
    test_filename = os.path.join("./data", "test/{:d}.mat".format(subject_id))

    # Create the dataset
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename,
        test_filename=test_filename,
        low_cut_hz=low_cut_hz,
    )

    # Load and normalize the data.
    train_set_x, mean, std = normalize(train_set.X)
    valid_set_x, _, _ = normalize(valid_set.X, mean, std)
    test_set_x, _, _ = normalize(test_set.X, mean, std)

    # Split the data into arrays.
    train_size = train_set.X.shape[0]
    train_input = np.array_split(train_set_x, train_size // batch_size)
    train_labels = np.array_split(train_set.y, train_size // batch_size)

    # TODO: Set a key using jax.random.PRNGKey
    # TODO: Create a cnn object by calling BrainCNN()
    cnn = None  # TODO: Store a model here.
    # TODO: Create an optimizer by calling optax.adam.

    # TODO: Initialize your model by calling cnn.init
    # TODO: A suitable input is jnp.expand_dims(jnp.ones(train_set.X.shape).transpose(0, 2, 1), -3).
    variables = None  # TODO: store actual variables here.

    # TODO: Initialize the optimizer by calling opt.init.

    # TODO: Make gradient computations possible by wrapping the forward_pass in
    # TODO: jax.value_and_grad.

    val_acc_list = []  # Use append to store validation accuracies here.
    for e in range(epochs):
        bar = tqdm(
            zip(train_input, train_labels),
            total=len(train_input),
            desc="Training Brain CNN",
        )
        for input_x, labels_y in bar:
            input_x = adjust_dimensions(input_x)
            labels_y = jnp.array(labels_y)

            # TODO: compute cel and the gradients using your wrapped forward pass.
            cel = 0.0  # TODO: Ensure cel contains the current cross entropy loss value.

            # TODO: get new updates and an optimizer state by calling your optimizers update function.

            # TODO: Apply the updates by calling optax.apply_updates.
            bar.set_description("Loss: {:2.3f}".format(cel))

        val_accuracy = get_acc(cnn, variables, valid_set_x, valid_set.y)
        print("Validation accuracy {:2.3f} at epoch {}".format(val_accuracy, e))  # type: ignore
        val_acc_list.append(val_accuracy)

    test_accuracy = get_acc(cnn, variables, test_set_x, test_set.y)
    print("Test accuracy: {:2.3f}".format(test_accuracy))  # type: ignore
    plt.plot(val_acc_list, label="Validation accuracy")
    plt.plot(len(val_acc_list) - 1, test_accuracy, ".", label="Test accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
