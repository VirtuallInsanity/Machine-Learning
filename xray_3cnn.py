
import numpy as np
import os
#import cv2

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.optimizers import RMSprop, SGD, Adam

from keras.callbacks import EarlyStopping

import gradcamutils
plt.rcParams['figure.figsize'] = 8,8

from keras.preprocessing.image import load_img, img_to_array

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix


TRAIN_DIR = 'chest_xray/train/'
TEST_DIR = 'chest_xray/test/'
VAL_DIR = 'chest_xray/val/'
input_path = 'chest_xray/'

#check for number of images
for set in ['train', 'val', 'test']:
    n_normal = len(os.listdir(input_path + set + '/normal'))
    n_infect = len(os.listdir(input_path + set + '/pneumonia'))
    print('Set: {}, normal images: {}, pneumonia images: {}'.format(set, n_normal, n_infect))

ROWS = 224#256
COLS = 224#256
CHANNELS = 1
batch_size = 32
epochs = 10

# make labels for test data
test_labels = []
for img in os.listdir(TEST_DIR + 'normal'):
  test_labels.append(0)
for img in os.listdir(TEST_DIR + 'pneumonia'):
  test_labels.append(1)
print('First label is: ', test_labels[0]) #0=normal 1=pneumonia
print("Test labels: ",len(test_labels))

# augmentation
test_aug = ImageDataGenerator(rescale=1./255)
train_aug = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   rotation_range=10)
test_aug = ImageDataGenerator(rescale = 1./255)  #Image normalization.


train_set = train_aug.flow_from_directory('chest_xray/train',
                                                 target_size = (ROWS, COLS),
                                                 batch_size = batch_size,
                                                 color_mode='grayscale',
                                                 class_mode = 'binary')
validation_set = test_aug.flow_from_directory('chest_xray/val/',
                                                        target_size=(ROWS, COLS),
                                                        batch_size=batch_size,
                                                        color_mode='grayscale',
                                                        class_mode='binary')
test_set = test_aug.flow_from_directory('chest_xray/test',
                                            target_size = (ROWS, COLS),
                                            batch_size = batch_size,
                                            color_mode='grayscale',
                                            class_mode = 'binary',
                                            shuffle=False)

print(train_set.class_indices)

# models

# №3
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# 3/3
# ----------Adam(lr=0.0001, decay=1e-6)----------
# loss: 0.3491 - accuracy: 0.8403 - val_loss: 0.6779 - val_accuracy: 0.7500
# Test loss: 0.660597562789917
# Test accuracy: 0.8541666865348816
# Predicted class: 0
# First prediction: [0.4823436]
#               precision    recall  f1-score   support
#
#       normal       0.81      0.80      0.80       234
#    pneumonia       0.88      0.89      0.88       390
#
#     accuracy                           0.85       624
#    macro avg       0.84      0.84      0.84       624
# weighted avg       0.85      0.85      0.85       624
#
#
#  --------------------  TEST METRICS --------------------
# 	Accuracy: 85.41666666666666%
# 	Precision: 88.04071246819338%
# 	Recall: 88.71794871794872%
# 	F1-score: 88.3780332056194
#
# 30/30
# ----------RMSprop(lr=0.001)----------------
# loss: 0.1416 - accuracy: 0.9551 - val_loss: 0.8667 - val_accuracy: 0.6250
# Test loss: 0.020599450916051865
# Test accuracy: 0.8125
# Predicted class: 0
# First prediction: [0.0427348]
#               precision    recall  f1-score   support
#
#       normal       0.95      0.53      0.68       234
#    pneumonia       0.78      0.98      0.87       390
#
#     accuracy                           0.81       624
#    macro avg       0.86      0.76      0.77       624
# weighted avg       0.84      0.81      0.80       624
#
#
#  --------------------  TEST METRICS --------------------
# 	Accuracy: 81.25%
# 	Precision: 77.57575757575758%
# 	Recall: 98.46153846153847%
# 	F1-score: 86.77966101694915

# №2
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

#3/3
# ----------Adam(lr=0.0001, decay=1e-6)----------
# loss: 0.2507 - accuracy: 0.8949 - val_loss: 0.5827 - val_accuracy: 0.6250
# Test loss: 0.5250895023345947
# Test accuracy: 0.8413461446762085
# Predicted class: 0
# First prediction: [0.33799577]
#               precision    recall  f1-score   support
#
#       normal       0.78      0.79      0.79       234
#    pneumonia       0.88      0.87      0.87       390
#
#     accuracy                           0.84       624
#    macro avg       0.83      0.83      0.83       624
# weighted avg       0.84      0.84      0.84       624
#
#
#  --------------------  TEST METRICS --------------------
# 	Accuracy: 84.13461538461539%
# 	Precision: 87.59689922480621%
# 	Recall: 86.92307692307692%
# 	F1-score: 87.25868725868725
#
# ----------RMSprop(lr=0.001)----------------
# 10/10
# loss: 0.1309 - accuracy: 0.9488 - val_loss: 0.5138 - val_accuracy: 0.6875
# Test loss: 0.02478131651878357
# Test accuracy: 0.8092948794364929
# Predicted class: 0
# First prediction: [0.16977352]
#               precision    recall  f1-score   support
#
#       normal       0.98      0.50      0.66       234
#    pneumonia       0.77      0.99      0.87       390
#
#     accuracy                           0.81       624
#    macro avg       0.88      0.75      0.76       624
# weighted avg       0.85      0.81      0.79       624
#
#
#  --------------------  TEST METRICS --------------------
# 	Accuracy: 80.92948717948718%
# 	Precision: 76.83168316831683%
# 	Recall: 99.48717948717949%
# 	F1-score: 86.70391061452513

# №1
model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# ----------Adam(lr=0.0001, decay=1e-6)----------
# 10/10
# loss: 0.1019 - accuracy: 0.9605 - val_loss: 0.3626 - val_accuracy: 0.7500
# Test loss: 0.004170296713709831
# Test accuracy: 0.8846153616905212
# Predicted class: 0
# First prediction: [0.00657666]
#               precision    recall  f1-score   support
#
#       normal       0.95      0.73      0.83       234
#    pneumonia       0.86      0.98      0.91       390
#
#     accuracy                           0.88       624
#    macro avg       0.90      0.85      0.87       624
# weighted avg       0.89      0.88      0.88       624
#
#
#  --------------------  TEST METRICS --------------------
# 	Accuracy: 88.46153846153845%
# 	Precision: 85.8108108108108%
# 	Recall: 97.6923076923077%
# 	F1-score: 91.36690647482013
#
# ----------RMSprop(lr=0.001)----------------
# 10/10
# loss: 0.1058 - accuracy: 0.9611 - val_loss: 0.1044 - val_accuracy: 1.0000
# Test loss: 0.010591854341328144
# Test accuracy: 0.9006410241127014
# Predicted class: 0
# First prediction: [0.00056508]
#               precision    recall  f1-score   support
#
#       normal       0.95      0.78      0.85       234
#    pneumonia       0.88      0.97      0.92       390
#
#     accuracy                           0.90       624
#    macro avg       0.91      0.88      0.89       624
# weighted avg       0.91      0.90      0.90       624
#
#
#  --------------------  TEST METRICS --------------------
# 	Accuracy: 90.06410256410257%
# 	Precision: 87.96296296296296%
# 	Recall: 97.43589743589743%
# 	F1-score: 92.45742092457421

print("input shape ",model.input_shape)
print("output shape ",model.output_shape)

model.summary()
#optimizer = RMSprop(lr=0.001)
#optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.7, nesterov=True)
optimizer = Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])


train_images=len(os.listdir(TRAIN_DIR+'normal/'))+len(os.listdir(TRAIN_DIR+'pneumonia/'))
print('Train img len',train_images)

#early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=2)
#, callbacks=[early_stopping]
history = model.fit_generator(generator=train_set, steps_per_epoch=train_images//batch_size, epochs=epochs, verbose=1, validation_data=validation_set)


scores = model.evaluate_generator(test_set)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = model.predict_generator(test_set)
predictions_classes = predictions.argmax(axis=1)
print("Predicted class:",predictions_classes[0])
print('First prediction:', predictions[0])

predict = np.round(model.predict_generator(test_set),0)
#print('rounded test\n',predict)
target = ['normal','pneumonia']
classification_metrics = classification_report(test_labels, predict, target_names=target)
print(classification_metrics)

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

# confution matrix
accuracy = accuracy_score(test_labels, np.round(predictions))*100
conf_mat = confusion_matrix(test_labels, np.round(predictions))
true_negative, false_postive, false_negative, true_posiitve = conf_mat.ravel()

#plt.title("CONFUSION MATRIX")
plot_confusion_matrix(conf_mat,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

print('\n','-'*20,' TEST METRICS', '-'*20)
precision = true_posiitve / (true_posiitve + false_postive) * 100
recall = true_posiitve / (true_posiitve + false_negative) * 100
print('\tAccuracy: {}%'.format(accuracy))
print('\tPrecision: {}%'.format(precision))
print('\tRecall: {}%'.format(recall))
print('\tF1-score: {}'.format(2*precision*recall/(precision+recall)))

#gradcam
# paths = ['test_normal.jpeg']#, 'test_normal_2.jpeg', 'test_normal_3.jpeg', 'test_pneumonia_virus.jpeg', 'test_pneumonia_virus_2.jpeg', 'test_pneumonia_bacteria.jpeg', 'test_pneumonia_bacteria_2.jpeg'
#
# for path in paths:
#     path = os.path.join("image", path)
#     orig_img = np.array(load_img(path, color_mode = "grayscale", target_size=(ROWS, COLS)), dtype=np.uint8)
#     img = np.expand_dims(np.array(load_img(path, color_mode = "grayscale", target_size=(ROWS, COLS)), dtype=np.float64), axis=-1)
#     img = np.expand_dims(img, axis=0)
#     #img = preprocess_input(img)
#     predictions = model.predict(img)
#     predictions_classes = predictions.argmax(axis=1)
#
#     gradcam = gradcamutils.grad_cam(model, img, layer_name='conv2d_2')#conv2d_2 for small model
#     gradcamplus = gradcamutils.grad_cam_plus(model, img, layer_name='conv2d_2')
#     print(path)
#     print("class activation map for:", predictions)
#     fig, ax = plt.subplots(nrows=1, ncols=3)
#     plt.subplot(131)
#     plt.imshow(orig_img)
#     plt.title("input image")
#     plt.subplot(132)
#     plt.imshow(orig_img)
#     plt.imshow(gradcam, alpha=0.8, cmap="jet")
#     plt.title("Grad-CAM")
#     plt.subplot(133)
#     plt.imshow(orig_img)
#     plt.imshow(gradcamplus, alpha=0.8, cmap="jet")
#     plt.title("Grad-CAM++")
#     plt.show()
