import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set_theme()

if __name__ == "__main__":
    m = np.array([[0,0]]).T
    cov = np.array([[1,0.5],[0.2,2]])

    #solution

    observations = []
    for i in range(1000):
        x = np.random.normal(size=(2,1))
        L = np.linalg.cholesky(cov)
        u = m + L@x
        observations.append(u)

    observations = np.concatenate(observations, axis=1)
    # Check the covariance matrix
    print(np.cov(observations))
    sns.jointplot(x=observations[0,:], y = observations[1,:], kind="kde", space=0, fill=True, aspect=1)
    plt.show()



        