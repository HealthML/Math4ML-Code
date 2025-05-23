import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA

def generate_3d_data(n=100):
    np.random.seed(42)
    mean = [0, 0, 0]
    cov = [[3, 1, 1],
           [1, 2, 1],
           [1, 1, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def SVD(X):
    # SVD is like matrix yoga — stretching into U, Σ, and V^T poses.

    # Step 1: Center data
    # Step 2: Compute covariance matrix C (C = X^T @ X)
    # Step 3: Eigen decomposition of C (Hint: np.linalg.eigh)
    # Step 4: Sort eigenvalues and eigenvectors descending 
    # Step 5: Calculate singular values (Sigma matrix values)
    # Step 6: Find Left singular vectors (U) using the formula U = X @ V @ Sigma_inv 
    # Return U, singular values, and V^T
    
    pass

def PCA(data, k=2):
    # Warning: PCA may cause dimensionality reduction addiction.

    # Step 1: Center the data
    # Step 2: Compute the covariance matrix of the centered data (Hint: np.cov)
    # Step 3: Compute eigenvalues and eigenvectors of the covariance matrix (use np.linalg.eigh)
    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    # Step 5: Select the first k eigenvectors (principal components) 
    # Step 6: Project the centered data onto the selected components (Hint: X @ components)
    # Return the projected data and the components ()
    pass


def visualize(data3d, projected):
    """Plot original 3D data and both 2D projections."""
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data3d[:,0], data3d[:,1], data3d[:,2], c='blue', alpha=0.6)
    ax1.set_title('Original 3D Data')
    ax2 = fig.add_subplot(122)
    ax2.scatter(projected[:,0], projected[:,1], c='red', alpha=0.6)
    ax2.set_title('2D Projection with PCA')
    plt.tight_layout()
    plt.show()


def main():
    data = generate_3d_data()
    projected, components = PCA(data, k=2)
    # visualize(projected)
    visualize(data, projected)
    U, S, Vt = SVD(data)
    print("Explicit SVD singular values:", S)

    from numpy.linalg import svd
    U_np, S_np, Vt_np = svd(data - np.mean(data, axis=0), full_matrices=False)
    print("Numpy SVD singular values:", S_np)

    svd_close = np.allclose(S, S_np, atol=1e-6)
    print("Singular values close:", svd_close)

    pca = skPCA(n_components=2)
    proj_sk = pca.fit_transform(data)

    # Compare PCA components using abs (ignore sign)
    comp_close_0 = np.allclose(np.abs(components[:, 0]), np.abs(pca.components_[0]), atol=1e-6)
    comp_close_1 = np.allclose(np.abs(components[:, 1]), np.abs(pca.components_[1]), atol=1e-6)

    print("PCA component 0 close to sklearn (ignoring sign):", comp_close_0)
    print("PCA component 1 close to sklearn (ignoring sign):", comp_close_1)

if __name__ == "__main__":
    main()
