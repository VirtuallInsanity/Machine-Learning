
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization
#from keras.optimizers import SGD, RMSprop
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
#import itertools
from keras.callbacks import EarlyStopping
from sklearn import metrics

train = pd.read_csv('venv/digits/train.csv')
test = pd.read_csv('venv/digits/test.csv')

print(train.shape)
print(test.shape)

batch_size = 128
#separate label
y_train = train['label']#only label .astype('int32')
x_train = train.loc[:,'pixel0':'pixel783']# label

#Normalize the data
x_train = x_train/255.0
test = test/255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
x_train = x_train.values.reshape(-1,28,28,1)
x_test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = to_categorical(y_train)
y_train.shape

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

aug = ImageDataGenerator(rotation_range=15, zoom_range=0.1,
	width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2)

train_batches = aug.flow(x_train, y_train, batch_size=batch_size)
val_batches = aug.flow(x_val, y_val, batch_size=batch_size)

#4 batch normalization at conv layer and dense layer
model = Sequential()
model.add(Convolution2D(32, (3, 3), padding="same", use_bias=False, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(64, (3, 3), padding="same", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(128, (3, 3), padding="same", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Activation("relu"))
model.add(Dense(10, use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Activation("softmax"))

print("input shape ",model.input_shape)
print("output shape ",model.output_shape)

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.7, nesterov=True)
#RMSprop(lr=0.001)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit_generator(generator=train_batches, steps_per_epoch=x_train.shape[0] // batch_size, epochs=12, verbose=1, validation_data=val_batches, callbacks=[early_stopping])


predict = model.predict_classes(x_test, verbose=0)
print('rounded test\n',predict)
predictions = model.predict(x_test)
predictions_classes = predictions.argmax(axis=1)
print("Predicted class:",predictions_classes[0])
print('First prediction:', predictions[0])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
# Loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation set', 'Training set'], loc='upper left')
plt.show()

#submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
#                         "Label": predictions})
#submissions.to_csv("minist_submission_batch_everywhere.csv", index=False, header=True)