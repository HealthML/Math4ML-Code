import numpy as np
from pca_exercise import PCA

# Example usage:
if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('./../datasets/breast_cancer_data/data_processed.csv')
    print(data.shape)
    # y includes our labels and x includes our features
    y = data.diagnosis      # M or B 
    list = ['diagnosis']
    df = data.drop(list,axis = 1 )  # load data into a dataframe
    X = df.values   # convert to a numpy array

    pca = PCA()
    pca.fit(X=X)

    X_pc = pca.transform(X)
    X_reconstruction_full = pca.reverse_transform(X_pc)
    print("L1 reconstruction error for full PCA : %.4E " % (np.absolute(X - X_reconstruction_full).sum()))

    for rank in range(X_pc.shape[1]+1):
        pca_lowrank = PCA(k=rank)
        pca_lowrank.fit(X=X)
        X_lowrank = pca_lowrank.transform(X)
        X_reconstruction = pca_lowrank.reverse_transform(X_lowrank)
        print("L1 reconstruction error for rank %i PCA : %.4E " % (rank, np.absolute(X - X_reconstruction).sum()))

    plt.ion()
    fig = plt.figure()
    plt.plot(X_pc[y=="M"][:,0], X_pc[y=="M"][:,1],'.', alpha = 0.3)
    plt.plot(X_pc[y=="B"][:,0], X_pc[y=="B"][:,1],'.', alpha = 0.3)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(["malignant", "benign"])

    fig2 = plt.figure()
    plt.plot(X_pc[y=="M"][:,0], X_pc[y=="M"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="B"][:,0], X_pc[y=="B"][:,2],'.', alpha = 0.3)
    plt.xlabel("PC 1")
    plt.ylabel("PC 3")
    plt.legend(["malignant", "benign"])


    fig3 = plt.figure()
    plt.plot(X_pc[y=="M"][:,1], X_pc[y=="M"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="B"][:,1], X_pc[y=="B"][:,2],'.', alpha = 0.3)
    plt.xlabel("PC 2")
    plt.ylabel("PC 3")
    plt.legend(["malignant", "benign"])

    fig4 = plt.figure()
    plt.plot(pca.variance_explained(),'.-')
    plt.xlabel("PC dimension")
    plt.ylabel("variance explained")

    fig4 = plt.figure()
    plt.plot(pca.variance_explained().cumsum() / pca.variance_explained().sum(),'.-')
    plt.xlabel("PC dimension")
    plt.ylabel("cumulative fraction of variance explained")

