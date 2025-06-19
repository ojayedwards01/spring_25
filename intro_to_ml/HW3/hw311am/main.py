# """Main script for the solution."""

# import numpy as np
# import pandas as pd
# import argparse

# import npnn


# def _get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--lr", help="learning rate", type=float, default=0.1)
#     p.add_argument("--opt", help="optimizer", default="SGD")
#     p.add_argument(
#         "--epochs", help="number of epochs to train", type=int, default=20)
#     p.add_argument(
#         "--save_stats", help="Save statistics to file", action="store_true")
#     p.add_argument(
#         "--save_pred", help="Save predictions to file", action="store_true")
#     p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
#     p.add_argument(
#         "--test_dataset", help="Dataset file (test set)",
#         default="mnist_test.npz")
#     p.set_defaults(save_stats=False, save_pred=False)
#     return p.parse_args()


# if __name__ == '__main__':
#     args = _get_args()
#     X, y = npnn.load_mnist(args.dataset)

#     # TODO
#     p = np.random.permutation(len(y))
#     X = X[p]
#     y = y[p]

#     train = npnn.Dataset(X[:50000], y[:50000], batch_size=32)
#     val = npnn.Dataset(X[50000:], y[50000:], batch_size=32)
    
#     # Create model (see npnn/model.py)
#     # Train for args.epochs
#     model = None
#     stats = pd.DataFrame()

#     # Save statistics to file.
#     # We recommend that you save your results to a file, then plot them
#     # separately, though you can also place your plotting code here.
#     if args.save_stats:
#         stats.to_csv("data/{}_{}.csv".format(args.opt, args.lr))

#     # Save predictions.
#     if args.save_pred:
#         X_test, _ = npnn.load_mnist("mnist_test.npz")
#         y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
#         np.save("mnist_test_pred.npy", y_pred)

"""Main training script."""

import numpy as np
import pandas as pd
import argparse
import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument("--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument("--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument("--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument("--test_dataset", help="Dataset file (test set)", default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


def train_model(args):
    """Train model with specific learning rate and return stats and model."""
    X, y = npnn.load_mnist(args.dataset)
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    
    train = npnn.Dataset(X[:50000], y[:50000], batch_size=32)
    val = npnn.Dataset(X[50000:], y[50000:], batch_size=32)

    modules = [
        npnn.Flatten(),
        npnn.Dense(784, 256),
        npnn.ELU(alpha=0.9),
        npnn.Dense(256, 64),
        npnn.ELU(alpha=0.9),
        npnn.Dense(64, 10)
    ]
    loss = npnn.SoftmaxCrossEntropy()
    
    if args.opt.lower() == "sgd":
        optimizer = npnn.SGD(learning_rate=args.lr, clipnorm=5.0)
    else:
        optimizer = npnn.Adam(learning_rate=args.lr)
    
    model = npnn.Sequential(modules, loss=loss, optimizer=optimizer)
    
    stats = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(args.epochs):
        train_loss, train_acc = model.train(train)
        val_loss, val_acc = model.test(val)
        
        stats['epoch'].append(epoch)
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
              f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")
    
    return stats, model


def evaluate_test_set(model, args):
    """Evaluate on test set and save predictions."""
    X_test, _ = npnn.load_mnist(args.test_dataset)
    y_pred = np.argmax(model.forward(X_test, train=False), axis=1).astype(np.uint8)
    np.save("mnist_test_pred.npy", y_pred)
    print("Test predictions saved in mnist_test_pred.npy")


if __name__ == '__main__':
    args = _get_args()
    stats, model = train_model(args)
    
    if args.save_stats:
        pd.DataFrame(stats).to_csv(f"data/{args.opt}_lr{args.lr}.csv", index=False)
    
    if args.save_pred:
        evaluate_test_set(model, args)