from cgi import test
from pickletools import optimize
from warnings import filters
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_dataset = pd.read_csv("./train.csv")
test_dataset = pd.read_csv("./test.csv")


X_train = train_dataset.loc[:, ~train_dataset.columns.isin(["label"])]
y_train = train_dataset["label"]

X_train = X_train/255.0
test_dataset = test_dataset/255.0



X_train = X_train.values.reshape(-1,28,28,1)
test_dataset = test_dataset.values.reshape(-1,28,28,1)


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state = 4   )

#print(X_train.shape)
#g = plt.imshow(X_train[5])
#g = plt.imshow(X_train[0][:,:,0])
#plt.show()


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = (5,5), padding='Same', activation="relu", input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = (5,5), padding='Same', activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = (3,3), padding='Same', activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = (3,3), padding='Same', activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-08,
    decay=0.0)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_acc",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=0.00001
)


model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 1
batch_size = 86


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, rotation_range=10, width_shift_range=0.1,
    height_shift_range=1.0, zoom_range=0.1,
    horizontal_flip=False, vertical_flip=False)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])



results = model.predict(test_dataset)
results = np.argmax(results,axis = 1)

print(results)