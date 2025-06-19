# """Neural Network model."""

# from .modules import Module
# from .optimizer import Optimizer

# import numpy as np


# def categorical_cross_entropy(pred, labels, epsilon=1e-10):
#     """Cross entropy loss function.

#     Parameters
#     ----------
#     pred : np.array
#         Softmax label predictions. Should have shape (dim, num_classes).
#     labels : np.array
#         One-hot true labels. Should have shape (dim, num_classes).
#     epsilon : float
#         Small constant to add to the log term of cross entropy to help
#         with numerical stability.

#     Returns
#     -------
#     float
#         Cross entropy loss.
#     """
#     assert(np.shape(pred) == np.shape(labels))
#     return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


# def categorical_accuracy(pred, labels):
#     """Accuracy statistic.

#     Parameters
#     ----------
#     pred : np.array
#         Softmax label predictions. Should have shape (dim, num_classes).
#     labels : np.array
#         One-hot true labels. Should have shape (dim, num_classes).

#     Returns
#     -------
#     float
#         Mean accuracy in this batch.
#     """
#     assert(np.shape(pred) == np.shape(labels))
#     return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


# class Sequential:
#     """Sequential neural network model.

#     Parameters
#     ----------
#     modules : Module[]
#         List of modules; used to grab trainable weights.
#     loss : Module
#         Final output activation and loss function.
#     optimizer : Optimizer
#         Optimization policy to use during training.
#     """

#     def __init__(self, modules, loss=None, optimizer=None):

#         for module in modules:
#             assert(isinstance(module, Module))
#         assert(isinstance(loss, Module))
#         assert(isinstance(optimizer, Optimizer))

#         self.modules = modules
#         self.loss = loss

#         self.params = []
#         for module in modules:
#             self.params += module.trainable_weights

#         self.optimizer = optimizer
#         self.optimizer.initialize(self.params)

#     def forward(self, X, train=True):
#         """Model forward pass.

#         Parameters
#         ----------
#         X : np.array
#             Input data

#         Keyword Args
#         ------------
#         train : bool
#             Indicates whether we are training or testing.

#         Returns
#         -------
#         np.array
#             Batch predictions; should have shape (batch, num_classes).
#         """
#         raise NotImplementedError()

#     def backward(self, y):
#         """Model backwards pass.

#         Parameters
#         ----------
#         y : np.array
#             True labels.
#         """
#         raise NotImplementedError()

#     def train(self, dataset):
#         """Fit model on dataset for a single epoch.

#         Parameters
#         ----------
#         X : np.array
#             Input images
#         dataset : Dataset
#             Training dataset with batches already split.

#         Notes
#         -----
#         You may find tqdm, which creates progress bars, to be helpful:

#         Returns
#         -------
#         (float, float)
#             [0] Mean train loss during this epoch.
#             [1] Mean train accuracy during this epoch.
#         """
#         raise NotImplementedError()

#     def test(self, dataset):
#         """Compute test/validation loss for dataset.

#         Parameters
#         ----------
#         dataset : Dataset
#             Validation dataset with batches already split.

#         Returns
#         -------
#         (float, float)
#             [0] Mean test loss.
#             [1] Test accuracy.
#         """
#         raise NotImplementedError()


"""Neural Network model."""

from .modules import Module
from .optimizer import Optimizer
import numpy as np


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function."""
    assert np.shape(pred) == np.shape(labels)
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic."""
    assert np.shape(pred) == np.shape(labels)
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model."""

    def __init__(self, modules, loss=None, optimizer=None):
        for module in modules:
            assert isinstance(module, Module)
        assert isinstance(loss, Module)
        assert isinstance(optimizer, Optimizer)

        self.modules = modules
        self.loss = loss
        self.params = []
        for module in modules:
            self.params += module.trainable_weights
        self.optimizer = optimizer
        self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass."""
        out = X
        for module in self.modules:
            out = module.forward(out, train=train)
        pred = self.loss.forward(out, train=train)
        return pred

    def backward(self, y):
        """Model backwards pass."""
        grad = self.loss.backward(y)
        for module in reversed(self.modules):
            grad = module.backward(grad)

    def train(self, dataset):
        """Fit model on dataset for a single epoch."""
        total_loss = 0.0
        total_acc = 0.0
        count = 0

        for X_batch, y_batch in dataset:
            pred = self.forward(X_batch, train=True)
            loss = categorical_cross_entropy(pred, y_batch)
            acc = categorical_accuracy(pred, y_batch)
            total_loss += loss
            total_acc += acc

            self.backward(y_batch)
            self.optimizer.apply_gradients(self.params)
            count += 1

        return total_loss / count, total_acc / count

    def test(self, dataset):
        """Compute test/validation loss for dataset."""
        total_loss = 0.0
        total_acc = 0.0
        count = 0

        for X_batch, y_batch in dataset:
            pred = self.forward(X_batch, train=False)
            loss = categorical_cross_entropy(pred, y_batch)
            acc = categorical_accuracy(pred, y_batch)
            total_loss += loss
            total_acc += acc
            count += 1

        return total_loss / count, total_acc / count