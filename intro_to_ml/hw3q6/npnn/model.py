"""Neural Network model."""

from .modules import Module
from .optimizer import Optimizer

import numpy as np


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).
    epsilon : float
        Small constant to add to the log term of cross entropy to help
        with numerical stability.

    Returns
    -------
    float
        Cross entropy loss.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).

    Returns
    -------
    float
        Mean accuracy in this batch.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model.

    Parameters
    ----------
    modules : Module[]
        List of modules; used to grab trainable weights.
    loss : Module
        Final output activation and loss function.
    optimizer : Optimizer
        Optimization policy to use during training.
    """

    def __init__(self, modules, loss=None, optimizer=None):

        # for module in modules:
        #     assert(isinstance(module, Module))
        # assert(isinstance(loss, Module))
        # assert(isinstance(optimizer, Optimizer))
        for module in modules:
            assert isinstance(module, Module)
        # Only assert loss if provided; allow None since main.py handles loss externally
        if loss is not None:
            assert isinstance(loss, Module)
        if optimizer is not None:
            assert isinstance(optimizer, Optimizer)

        self.modules = modules
        self.loss = loss

        self.params = []
        for module in modules:
            self.params += module.trainable_weights

        self.optimizer = optimizer
        self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass.

        Parameters
        ----------
        X : np.array
            Input data

        Keyword Args
        ------------
        train : bool
            Indicates whether we are training or testing.

        Returns
        -------
        np.array
            Batch predictions; should have shape (batch, num_classes).
        """
        # raise NotImplementedError()
        output = X
        for module in self.modules:
            output = module.forward(output, train=train)
        return output

    def backward(self, y):
        """Model backwards pass.

        Parameters
        ----------
        y : np.array
            True labels.
        """
        # raise NotImplementedError()
        input_grad = grad
        for module in reversed(self.modules):
            input_grad = module.backward(input_grad)
        return input_grad

    def train(self, dataset):
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        X : np.array
            Input images
        dataset : Dataset
            Training dataset with batches already split.

        Notes
        -----
        You may find tqdm, which creates progress bars, to be helpful:

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        # raise NotImplementedError()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch_X, batch_y in dataset:
            # Forward pass
            y_pred = self.forward(batch_X, train=True)
            loss = self.loss.forward(y_pred, batch_y)
            total_loss += loss
            total_accuracy += categorical_accuracy(y_pred, batch_y)
            num_batches += 1

            # Backward pass
            grad = self.loss.backward(batch_y)
            self.backward(grad)
            self.optimizer.apply_gradients(self.params)

            # Zero gradients
            for param in self.params:
                param.grad = None

        return total_loss / num_batches, total_accuracy / num_batches

    def test(self, dataset):
        """Compute test/validation loss for dataset.

        Parameters
        ----------
        dataset : Dataset
            Validation dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean test loss.
            [1] Test accuracy.
        """
        # raise NotImplementedError()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch_X, batch_y in dataset:
            y_pred = self.forward(batch_X, train=False)
            loss = self.loss.forward(y_pred, batch_y)
            total_loss += loss
            total_accuracy += categorical_accuracy(y_pred, batch_y)
            num_batches += 1

        return total_loss / num_batches, total_accuracy / num_batches
