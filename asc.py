#---------------------------------------
#Since : 2018/06/01
#Update: 2021/06/15
# -*- coding: utf-8 -*-
#---------------------------------------

import numpy as np
import math as mt
import pylab as plt
import sys
from scipy import ndimage
from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from sklearn import metrics

from ng import NG
from gng import GNG
from kohonen_2d import Kohonen

class SomSpectralClustering:
    def __init__(self, x, k = 3, n = 64, sigma = 1, lap = "Lsym", som = "NG"):
        # normalized data points
        self.x = self.normalize(x)
        # the number of data points
        self.num = x.shape[0]
        # the number of clusters
        self.k = k
        # the number of units
        self.n = n
        # the type of Laplacian matrix
        self.lap = lap
        # the method of SOMs
        self.som = som

    # normalize data points
    def normalize(self, data):
        norm_data = np.linalg.norm(data, axis=1)
        max_norm = np.linalg.norm(data[np.argmax(norm_data)])
        data /= max_norm
        return data

    # calculate normalized Laplacian matrix
    def Laplacian(self, affinity):
        # calculate Laplacian matrix
        degree = np.diag(np.sum(affinity, axis = 0))
        L = degree - affinity

        # normalize the Laplacian matrix
        if self.lap == "Lrw":
            L = np.dot(np.linalg.inv(degree), L)
        elif self.lap == "Lsym":
            degree = np.linalg.inv(np.diag(np.sqrt(np.sum(affinity, axis = 0))))
            L = np.dot(np.dot(degree, L), degree)

        return L

    def fit(self):
        if self.som == "GNG":
            aff_net = GNG(num = self.n, end = 100000, lam = 250, ew = 0.1, en = 0.01, amax = 75, alpha = 0.25,  beta = 0.99, sig_kernel = 0.25)
        elif self.som == "Kohonen":
            aff_net = Kohonen(num = self.n, dim = self.x.shape[1], end = 100000, rate = 0.05, sigma = 1.0, sig_kernel = 0.5)
        elif self.som == "NG":
            aff_net = NG(num = self.n, end = 100000, lam_i = 1.0, lam_f = 0.01, ew_i = 0.5, ew_f = 0.005, amax_i = 100.0, amax_f = 300.0, sig_kernel = 0.25)

        # make the affinity matrix using SOM
        aff_net.train(self.x)

        # save the affinity matrix
        self.affinity = aff_net.affinity()

        # calculate Laplacian matrix and its eigen vectors
        self.L = self.Laplacian(self.affinity)
        eig_val, eig_vec = np.linalg.eig(self.L)
        eig_vec = eig_vec.real

        # assign the units to the clusters
        kmeans = KMeans(n_clusters=self.k)
        self.labels_units = kmeans.fit(eig_vec[:,np.argsort(eig_val)[0:self.k]]).labels_

        self.units = aff_net.units

        # assing the data points to the clusters
        self.labels = np.zeros(self.num)
        for i in range(self.num):
            distances =  np.linalg.norm(self.units - self.x[i], axis = 1)
            self.labels[i]= self.labels_units[np.argmin(distances)]

        return self.labels


if __name__ == '__main__':

    data, true_labels = datasets.make_blobs(n_samples=1000, centers=3)
    ssc = SomSpectralClustering(som = "GNG", x = data, k = 3, n = 100)
    labels = ssc.fit()
    print(metrics.adjusted_rand_score(true_labels, labels))
