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


"""Main script for the solution."""

import numpy as np
import pandas as pd
import argparse

import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


if __name__ == '__main__':
    args = _get_args()
    X, y = npnn.load_mnist(args.dataset)  # X: (n, 28, 28), y: (n, 10)

    # Shuffle data
    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]

    # Split: 50,000 train, rest val; batch_size=32 gives 1562 batches (49,984 samples)
    train = npnn.Dataset(X[:50000], y[:50000], batch_size=32)
    val = npnn.Dataset(X[50000:], y[50000:], batch_size=32)
    
    # Create model (256-64-10 architecture from problem 6.4)
    model = npnn.Sequential([
        npnn.Flatten(),              # (32, 28, 28) -> (32, 784)
        npnn.Dense(784, 256),        # Input: 784, Hidden: 256
        npnn.ELU(),                  # Activation
        npnn.Dense(256, 64),         # Hidden: 64
        npnn.ELU(),                  # Activation
        npnn.Dense(64, 10),          # Output: 10 classes
    ])
    
    # Define optimizer
    if args.opt == "SGD":
        optimizer = npnn.SGD(model.parameters(), learning_rate=args.lr)
    elif args.opt == "Adam":
        optimizer = npnn.Adam(model.parameters(), learning_rate=args.lr)
    else:
        raise ValueError(f"Optimizer {args.opt} not supported")
    
    optimizer.initialize(model.parameters())  # Optional for SGD, needed for Adam
    
    # Loss function
    loss_fn = npnn.SoftmaxCrossEntropy()
    
    # Training loop (replaces TODO)
    stats = pd.DataFrame(columns=["epoch", "train_loss", "val_accuracy"])
    for epoch in range(args.epochs):
        train_loss = 0
        for batch_X, batch_y in train:  # batch_X: (32, 28, 28), batch_y: (32, 10)
            # Forward pass
            y_pred = model.forward(batch_X, train=True)  # (32, 10)
            loss = loss_fn.forward(y_pred, batch_y)
            train_loss += loss
            
            # Backward pass
            grad = loss_fn.backward(batch_y)  # Pass labels to get gradient
            model.backward(grad)
            optimizer.apply_gradients(model.parameters())
            
            # Zero gradients
            for param in model.parameters():
                param.grad = None
        
        # Validation accuracy
        val_correct = 0
        total = 0
        for batch_X, batch_y in val:
            y_pred = model.forward(batch_X, train=False)
            val_correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(batch_y, axis=1))
            total += len(batch_y)
        val_accuracy = val_correct / total
        
        # Record stats
        train_loss_avg = train_loss / len(train)
        stats.loc[epoch] = [epoch, train_loss_avg, val_accuracy]
        print(f"Epoch {epoch}: Train Loss = {train_loss_avg:.4f}, Val Accuracy = {val_accuracy:.4f}")

    # Save statistics to file
    if args.save_stats:
        stats.to_csv("data/{}_{}.csv".format(args.opt, args.lr))

    # Save predictions
    if args.save_pred:
        X_test, _ = npnn.load_mnist("mnist_test.npz")
        y_pred = np.argmax(model.forward(X_test, train=False), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)