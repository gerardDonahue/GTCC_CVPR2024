import numpy as np
from sklearn.decomposition import PCA


def pca(data, n_components):
    """
        This function takes in data, and the desired number of 
            principle components for PCA
        Then returns the dataset with the lower dimensionality as   
            a result of PCA algorithm

        Inputs:
        - (data) numpy array dataset
        - (n_components) labels for this video (should be arbitrary for now...)

        Outputs:
            (dataset) - dataset in (original shape x n_components)
    """
    pca = PCA(n_components=n_components)

    # Fit the PCA model to the data
    pca.fit(data)

    # Transform the data to the first 10 principal components
    transformed_data = pca.transform(data)
    return transformed_data
