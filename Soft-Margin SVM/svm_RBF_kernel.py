import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
training_data = pd.read_csv('digits_training_data.csv', header=None)
training_labels = pd.read_csv('digits_training_labels.csv', header=None)
test_data = pd.read_csv('digits_test_data.csv', header=None)
test_labels = pd.read_csv('digits_test_labels.csv', header=None)

# Create the SVM with RBF kernel
svc = SVC(kernel='rbf', C=1.0, gamma='scale')

# Train the model
svc.fit(training_data, training_labels.values.ravel())

# Predictions
train_predictions = svc.predict(training_data)
test_predictions = svc.predict(test_data)

# Calculate accuracies
train_accuracy = accuracy_score(training_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Find misclassified images
misclassified_indices = np.where(test_predictions != test_labels.values.ravel())[0]
misclassified_images = test_data.iloc[misclassified_indices]
misclassified_labels = test_labels.iloc[misclassified_indices]
misclassified_predictions = test_predictions[misclassified_indices]

# Reshape the images to the correct size (26x26 since the training data has 676 features per image)
image_size = int(np.sqrt(training_data.shape[1]))

# Select up to 5 misclassified images to show
max_images_to_show = 5
images_to_show = min(len(misclassified_indices), max_images_to_show)

# Plot the misclassified images
fig, axes = plt.subplots(1, images_to_show, figsize=(10, 3))
for i in range(images_to_show):
    ax = axes[i] if images_to_show > 1 else axes
    ax.imshow(misclassified_images.iloc[i].values.reshape(image_size, image_size), cmap='gray')
    ax.set_title(f'Pred: {misclassified_predictions[i]}')
    ax.axis('off')
plt.show()

print(f'Training accuracy: {train_accuracy}')
print(f'Test accuracy: {test_accuracy}')
print(f'Model parameters: {svc.get_params()}')
print(f'Number of misclassified test images: {len(misclassified_indices)}')

# If you want to save the plot of misclassified images, uncomment the following line:
# plt.savefig('misclassified_digits.png')
