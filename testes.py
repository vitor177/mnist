from cgi import test
from pickletools import optimize
from warnings import filters
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


test_dataset = pd.read_csv("./test.csv")

# shape (28000, 784)
test_dataset = test_dataset/255.0

test_dataset = test_dataset.values.reshape(-1,28,28,1)

#print(X_train.shape)
g = plt.imshow(test_dataset[1])
#g = plt.imshow(test_dataset[0][:,:,0])
plt.show()

print(test_dataset.shape)

# 2 0 9 3 9 2