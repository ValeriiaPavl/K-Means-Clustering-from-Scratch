import numpy as np
import sklearn
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from build_clusters import CustomKMeans


# scroll down to the bottom to implement your solution

def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):
    # This function for visualizing the results of clustering

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


def find_best_K(X):
    prev_cluster = CustomKMeans(1)
    prev_cluster.fit(X)
    prev_error = prev_cluster.find_the_error(X)

    curr_cluster = CustomKMeans(2)
    curr_cluster.fit(X)
    curr_error = curr_cluster.find_the_error(X)

    errors_list = [prev_error, curr_error]
    best_k = 2

    while abs((curr_error - prev_error) / prev_error) > 0.2:
        best_k += 1
        cluster = CustomKMeans(best_k)
        cluster.fit(X)
        prev_error, curr_error = curr_error, cluster.find_the_error(X)
        errors_list.append(curr_error)

    return best_k - 1


if __name__ == '__main__':
    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = MinMaxScaler()
    X_full = scaler.fit_transform(X_full)
    best_size = find_best_K(X_full)
    clusters = CustomKMeans(best_size)
    clusters.fit(X_full)
    prediction = (clusters.predict(X_full[0:20]))
    print(prediction)
    # plot_comparison(X_full, clusters.predict(X_full))
