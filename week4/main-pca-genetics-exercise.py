# !pip install pysnptools
from pysnptools.snpreader import Bed
import numpy as np
from pca_exercise import PCA

# Example usage:
if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    snpreader = Bed('./../datasets/genetic_data/example2.bed', count_A1=True)
    data = snpreader.read()
    print(data.shape)
    # y includes our labels and x includes our features
    labels = pd.read_csv("./../datasets/genetic_data/1kg_annotations_edit.txt", sep="\t", index_col="Sample")
    list1 = data.iid[:,1].tolist()  #list with the Sample numbers present in genetic dataset
    labels = labels[labels.index.isin(list1)]  #filter labels DataFrame so it only contains the sampleIDs present in genetic data
    y = labels.SuperPopulation  # EUR, AFR, AMR, EAS, SAS
    X = data.val[:, ~np.isnan(data.val).any(axis=0)]  #load genetic data to X, removing NaN values

    pca = PCA()
    pca.fit(X=X)

    X_pc = pca.transform(X)
    X_reconstruction_full = pca.reverse_transform(X_pc)
    print("L1 reconstruction error for full PCA : %.4E " % (np.absolute(X - X_reconstruction_full).sum()))

    for rank in range(15):    #more correct: X_pc.shape[1]+1
        pca_lowrank = PCA(k=rank)
        pca_lowrank.fit(X=X)
        X_lowrank = pca_lowrank.transform(X)
        X_reconstruction = pca_lowrank.reverse_transform(X_lowrank)
        print("L1 reconstruction error for rank %i PCA : %.4E " % (rank, np.absolute(X - X_reconstruction).sum()))

    fig = plt.figure()
    plt.plot(X_pc[y=="EUR"][:,0], X_pc[y=="EUR"][:,1],'.', alpha = 0.3)
    plt.plot(X_pc[y=="AFR"][:,0], X_pc[y=="AFR"][:,1],'.', alpha = 0.3)
    plt.plot(X_pc[y=="EAS"][:,0], X_pc[y=="EAS"][:,1],'.', alpha = 0.3)
    plt.plot(X_pc[y=="AMR"][:,0], X_pc[y=="AMR"][:,1],'.', alpha = 0.3)
    plt.plot(X_pc[y=="SAS"][:,0], X_pc[y=="SAS"][:,1],'.', alpha = 0.3)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(["EUR", "AFR","EAS","AMR","SAS"])

    fig2 = plt.figure()
    plt.plot(X_pc[y=="EUR"][:,0], X_pc[y=="EUR"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="AFR"][:,0], X_pc[y=="AFR"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="EAS"][:,0], X_pc[y=="EAS"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="AMR"][:,0], X_pc[y=="AMR"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="SAS"][:,0], X_pc[y=="SAS"][:,2],'.', alpha = 0.3)
    plt.xlabel("PC 1")
    plt.ylabel("PC 3")
    plt.legend(["EUR", "AFR","EAS","AMR","SAS"])


    fig3 = plt.figure()
    plt.plot(X_pc[y=="EUR"][:,1], X_pc[y=="EUR"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="AFR"][:,1], X_pc[y=="AFR"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="EAS"][:,1], X_pc[y=="EAS"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="AMR"][:,1], X_pc[y=="AMR"][:,2],'.', alpha = 0.3)
    plt.plot(X_pc[y=="SAS"][:,1], X_pc[y=="SAS"][:,2],'.', alpha = 0.3)
    plt.xlabel("PC 2")
    plt.ylabel("PC 3")
    plt.legend(["EUR", "AFR","EAS","AMR","SAS"])

    fig4 = plt.figure()
    plt.plot(pca.variance_explained(),'.-')
    plt.xlabel("PC dimension")
    plt.ylabel("variance explained")

    fig4 = plt.figure()
    plt.plot(pca.variance_explained().cumsum() / pca.variance_explained().sum(),'.-')
    plt.xlabel("PC dimension")
    plt.ylabel("cumulative fraction of variance explained")
    plt.show()

