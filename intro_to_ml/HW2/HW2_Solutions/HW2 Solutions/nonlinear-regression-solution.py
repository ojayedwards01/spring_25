import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv, pinv
from sklearn.model_selection import train_test_split


class SinusoidalRegressor:
    def __init__(self):
        self.k = None
        self.weights = None

    def phi(self, x):
        # The basis function for a general 2k
        return np.array([1] + [func(i * x) for i in range(1, self.k + 1) for func in (np.sin, np.cos)])

    def fit(self, X_train, Y_train, k):
        self.k = k
        # Make sure Y_train is a one-dimensional array
        Y_train = Y_train.flatten()
        # Construct the design matrix Phi for all data points in X_train
        Phi = np.array([self.phi(x) for x in X_train])
        # Solve for the weights using the normal equation with a pseudo-inverse
        # Make sure the shapes align: Phi.T @ Phi should be a square matrix and Phi.T @ Y_train should be a vector
        self.weights = pinv(Phi.T @ Phi) @ Phi.T @ Y_train

    def predict(self, X):
        # Check if the model is fitted
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        # Apply the learned model
        return np.array([self.phi(x) @ self.weights for x in X])

    def rmse(self, X_val, Y_val):
        # Predict the values for X_val
        predictions = self.predict(X_val)
        # Calculate the RMSE
        return np.sqrt(np.mean((Y_val - predictions) ** 2))


np.random.seed(61)
csv_file = 'nonlinear-regression-data.csv'
data = pd.read_csv(csv_file)
x = np.array(data['X'])
y = np.array(data['Noisy_y'])


### Evaluation Part 0 #################################################################################

# Split the data
test_indices = np.random.choice(np.arange(0,60),16,replace=True)
X_val = []
Y_val = []
X_train = []
Y_train = []
for i in range(len(x)):
    if i in test_indices:
        X_val.append(x[i]) 
        Y_val.append(y[i])
    else:
        X_train.append(x[i])
        Y_train.append(y[i])
X_val = np.array(X_val)
X_train = np.array(X_train)
Y_val = np.array(Y_val)
Y_train = np.array(Y_train)


### Evaluation Part 1 and 2 #################################################################################
# Dictionary to store RMSE for each k
k_rmse_train = {}
# Dictionary to store RMSE for each k
k_rmse = {}

# Initialize the model
model = SinusoidalRegressor()

# Vary k from 1 to 10 and obtain RMSE error on the training set
for k in range(1, 11):
    model.fit(X_train, Y_train, k)

    rmse_value_train = model.rmse(X_train, Y_train)
    k_rmse_train[k] = rmse_value_train

    rmse_value = model.rmse(X_val, Y_val)
    k_rmse[k] = rmse_value

# Plotting the training error versus k
plt.figure(figsize=(10, 6))
plt.plot(list(k_rmse_train.keys()), list(k_rmse_train.values()), marker='o')
plt.title('Training RMSE Error vs. Polynomial Degree k')
plt.xlabel('Polynomial Degree k')
plt.ylabel('Training RMSE Error')
plt.grid(True)
plt.show()

# Plotting the validation error versus k
plt.figure(figsize=(10, 6))
plt.plot(list(k_rmse.keys()), list(k_rmse.values()), marker='o', color='red')
plt.title('Validation RMSE Error vs. Polynomial Degree k')
plt.xlabel('Polynomial Degree k')
plt.ylabel('Validation RMSE Error')
plt.grid(True)
plt.show()

### Evaluation Part 4 #################################################################################
# Create separate plots for each k
k_values = [1, 2, 4, 10]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Generate a range of x values for plotting the fitted polynomials
x_range = np.linspace(min(x), max(x), 100)

for i, k in enumerate(k_values):
    # Fit the model for the current k
    model.fit(X_train, Y_train, k)
    
    # Predict values using the fitted model
    y_range = model.predict(x_range)
    
    # Scatter plot of the validation data points on subplot i
    axs[i].scatter(X_val, Y_val, color='black', label='Validation Data')
    
    # Draw a line for the fitted polynomial on subplot i
    axs[i].plot(x_range, y_range, label=f'Fitted Polynomial (k={k})')
    
    # Set title and labels for subplot i
    axs[i].set_title(f'Fitted Polynomial with k={k}')
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('Y')
    axs[i].legend()
    axs[i].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()