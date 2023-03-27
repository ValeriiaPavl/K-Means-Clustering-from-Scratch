from math import sqrt

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
import scipy
from scipy.spatial import distance


class CustomKMeans:
    def __init__(self, k):
        self.k = k
        self.centers = None

    def fit(self, X, eps=1e-6):
        first_labels = self.find_nearest_centers(X, X[:self.k])
        first_centers = self.calculate_new_centers(X, first_labels)

        new_labels = self.find_nearest_centers(X, first_centers)
        self.centers = self.calculate_new_centers(X, new_labels)

        square_distances = [np.linalg.norm(old - new)
                            for old, new in zip(first_centers, self.centers)]

        while any([d > eps for d in square_distances]):
            new_labels = self.find_nearest_centers(X, self.centers)
            old_centers, self.centers = self.centers, self.calculate_new_centers(X, new_labels)
            square_distances = [np.linalg.norm(old - new) for old, new in zip(old_centers, self.centers)]

    def predict(self, X):
        return self.find_nearest_centers(X, self.centers)

    def find_the_error(self, X):  # the error for this k value
        labels = self.predict(X)
        sse_features = [self.find_feature_error(point, self.centers[label])
                        for point, label in zip(X, labels)]
        error = sqrt(sum(sse_features) / len(self.centers))
        return error

    @staticmethod
    def find_feature_error(feature, center):
        sse = mean_squared_error(feature, center) * len(center)  # sum of squared errors
        return sse

    @staticmethod
    def find_nearest_centers(features, *centroids):
        labels = []
        for point in features:
            nearest_distance = distance.euclidean(point, centroids[0][0])
            centroid_label = 0
            for label, centroid in enumerate(centroids[0]):
                e_distance = distance.euclidean(point, centroid)
                if e_distance < nearest_distance:
                    nearest_distance = e_distance
                    centroid_label = label
            labels.append(centroid_label)
        return labels

    @staticmethod
    def calculate_new_centers(features, labels):
        total_labels = max(set(labels)) + 1  # total amount of labels
        sorted_features = [[] for _ in range(total_labels)]  # creating empty lists for sorted features
        [sorted_features[label].append(feature)
         for label, feature in zip(labels, features)]  # sorting features
        new_cents = np.array(
            [np.mean(cluster, axis=0) for cluster in sorted_features])  # calculating means for every label
        return new_cents
