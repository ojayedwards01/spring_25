from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

def pca_fun(input_data, target_d):

    # Center the data
    mean = np.mean(input_data, axis=0)
    centered_data = input_data - mean

    # Covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    top_eigvecs = eigvecs[:, sorted_indices[:target_d]]

    return top_eigvecs


### Data loading and plotting the image ###
import os
print(os.path.exists('face_data.mat'))
data = loadmat('face_data.mat')
image = data['image'][0]
person_id = data['personID'][0]

plt.imshow(image[0], cmap='gray')
plt.show()