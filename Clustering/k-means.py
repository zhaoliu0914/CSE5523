from __future__ import division

import random
import numpy as np
from matplotlib import pyplot as plt

# Load the mandrill image as an NxNx3 array. Values range from 0.0 to 255.0.
mandrill = plt.imread('mandrill.png')[:,:,:3].astype(float)

N = int(mandrill.shape[0])

#print(mandrill.shape)

M = 2
k = 64

# Store each MxM block of the image as a row vector of X
X = np.zeros((N**2//M**2, 3*M**2))
for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)


# Implement k-means and cluster the rows of X,
# then reconstruct the compressed image using the cluster center for each block,
# as specified in the homework description.

#print(X)

# initialize the cluster mu by randomly selected data points
mus = np.zeros((k, 3*M**2))
for index in range(k):
    point_index = random.randint(0, (N**2//M**2) -1)
    mus[index] = X[point_index]


is_continue = True
iteration_count = 0

while is_continue:
    iteration_count += 1

    # Initialize cluster group for every mu
    cluster = dict()
    for index in range(k):
        cluster[index] = list()

    # Calculate the distance between every point to every mu
    # Cluster/Group every point to the smallest mu
    for point_index in range(X.shape[0]):
        distances = []
        point = X[point_index]

        for mu in range(mus.shape[0]):
            # Calculate the L2 distance between point and mu
            l2_distance = np.linalg.norm(point - mu)

            distances.append(l2_distance)

        # Find the index of the smallest distance
        mu_index = distances.index(min(distances))
        cluster[mu_index].append(point_index)

    # re-estimate cluster centers by averaging over points
    is_continue = False
    for index in range(k):
        mu_new = np.zeros(3*M**2)

        point_index_list = cluster[index]
        if len(point_index_list) > 0:
            for point_index in point_index_list:
                point = X[point_index]
                mu_new = mu_new + point
            #mu_new = np.multiply(mu_new, 1/len(point_index_list))
            mu_new = (1 / len(point_index_list)) * mu_new

            mu = mus[index]
            if np.array_equal(mu, mu_new) is False:
                is_continue = True
                mus[index] = mu_new


print(f"mus = {mus}")
print(f"iteration_count = {iteration_count}")

