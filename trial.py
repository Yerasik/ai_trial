import cv2
import numpy as np
import matplotlib.pyplot as plt

# Scan all the directories and create a list of labels
labels = os.listdir( 'fashion_mnist_images/train')
# Create lists for samples and labels
X = []
y = []
# For each label folder
for label in labels:
  # And for each image in given folder
  for file in os.listdir(os.path.join('fashion_mnist_images','train',label)):
    # Read the image
    image = cv2.imread(os.path.join('fashion_mnist_images/train', label, file), cv2.IMREAD_UNCHANGED)
    # And append it and a label to the lists
    X.append(image)
    y.append(label)

