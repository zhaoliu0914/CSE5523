import random
import numpy as np
import matplotlib.pyplot as plt


num = random.randrange(0, 100)
print(f"num = {num}")

"""
plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])
plt.show()

plt.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
plt.show()

"""

mandrill = plt.imread('Clustering/mandrill.png')[:,:,:3].astype(float)

for i in range(512):
    for j in range(512):
        for k in range(3):
            mandrill[i][j][k] = mandrill[i][j][k] + 128/255

plt.imshow(mandrill)
plt.show()


image = plt.imread("Image Generate/4_pixel_image.png")[:,:,:3].astype(float)

#plt.imshow(image)
#plt.show()
print(f"image.shape = {image.shape}")
print(f"image = {image}")

for i in range(2):
    for j in range(2):
        for k in range(3):
            image[i][j][k] = image[i][j][k] + 128/255

#neutral_gray = np.zeros(12).astype(float)
#for index in range(neutral_gray.shape[0]):
#    neutral_gray[index] = 128/255

#image = neutral_gray.reshape((2, 2, 3))
#print(f"image.shape = {image.shape}")
#print(f"image = {image}")

#plt.imshow(image)
#plt.show()