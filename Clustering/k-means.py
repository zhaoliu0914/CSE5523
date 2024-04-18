from __future__ import division

import random
import numpy as np
from matplotlib import pyplot as plt

import datetime
print(f"start time = {datetime.datetime.now()}")

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
        X[i*N//M+j, :] = mandrill[i*M:(i+1)*M, j*M:(j+1)*M, :].reshape(3*M**2)

X_copy = np.copy(X)

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
distance_list = []
iteration_list = []
point_mu_map = np.zeros((N**2//M**2, 2))
for point_index in range(point_mu_map.shape[0]):
    point_mu_map[point_index][0] = point_index

while is_continue:
    total_distances = 0
    iteration_count += 1

    last_point_mu_map = np.copy(point_mu_map)

    # Calculate the distance between every point to every mu
    # Cluster/Group every point to the smallest mu
    for point_index in range(X.shape[0]):
        distances = []
        point = X[point_index]

        for mu_index in range(mus.shape[0]):
            # Calculate the L2 distance between point and mu
            mu = mus[mu_index]
            l2_distance = np.linalg.norm(point - mu)
            l2_distance = l2_distance**2

            distances.append(l2_distance)

        # Find the index of the smallest distance
        #smallest_distance = min(distances)
        smallest_distance = np.min(distances)
        total_distances += smallest_distance
        #mu_index = distances.index(smallest_distance)
        mu_index = np.argmin(distances)
        point_mu_map[point_index][1] = mu_index

    if iteration_count == 20:
        is_continue = False
    #for index in range(point_mu_map.shape[0]):
    #    if last_point_mu_map[index][1] != point_mu_map[index][1]:
    #        print(f"last_point_mu_map[{index}][1] = {last_point_mu_map[index][1]}")
    #        print(f"point_mu_map[{index}][1] = {point_mu_map[index][1]}")
    #        is_continue = True

    # re-estimate cluster centers by averaging over points
    for index in range(mus.shape[0]):
        mu_new = np.zeros(3*M**2)

        point_index_list = []
        for point_index in range(point_mu_map.shape[0]):
            if point_mu_map[point_index][1] == index:
                point_index_list.append(point_index)
        if len(point_index_list) > 0:
            for point_index in point_index_list:
                point = X[point_index]
                mu_new = mu_new + point
            mu_new = (1 / len(point_index_list)) * mu_new

            mu = mus[index]
            if np.array_equal(mu, mu_new) is False:
                #is_continue = True
                mus[index] = mu_new

    print(f"iteration_count = {iteration_count}")
    print(f"total_distances = {total_distances}")
    distance_list.append(total_distances)
    iteration_list.append(iteration_count)

neutral_gray = np.zeros(3*M**2)
for index in range(neutral_gray.shape[0]):
    neutral_gray[index] = 128/255

for index in range(X_copy.shape[0]):
    mu_index = point_mu_map[index][1]
    #relative_mean = abs(mus[int(mu_index)] - X_copy[index])
    X_copy[index] = np.add(mus[int(mu_index)], neutral_gray)

#print(f"iteration_count = {iteration_count}")
#print(f"mus = {mus}")
#print(f"point_mu_map = {point_mu_map}")


plt.plot(iteration_list, distance_list)
plt.show()

mandrill_compressed = X_copy.reshape((512, 512, 3))
plt.imshow(mandrill_compressed)
plt.show()

relative_mean_absolute_error = 0
for i in range(N):
    for j in range(N):
        for a in range(3):
            relative_mean_absolute_error += abs(mandrill_compressed[i][j][a] - mandrill[i][j][a])
relative_mean_absolute_error = relative_mean_absolute_error / (3 * X_copy.shape[0]**2)

print(f"relative_mean_absolute_error = {relative_mean_absolute_error}")
