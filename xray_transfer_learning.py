import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import style
import seaborn as sns

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import random as rn

import cv2
import numpy as np
import os

from sklearn.metrics import accuracy_score, confusion_matrix

import gradcamutils
plt.rcParams['figure.figsize'] = 8,8

from scipy.ndimage.interpolation import zoom
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.applications.vgg16 import VGG16

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

TRAIN_DIR = 'chest_xray/train/'#'venv/data_xray2/train/'
TEST_DIR = 'chest_xray/test/'#'venv/data_xray2/test/'
VAL_DIR = 'chest_xray/val/' #'venv/data_xray2/val/'
input_path = 'chest_xray/'#'venv/data_xray2/

ROWS = 224
COLS = 224
CHANNELS = 3

# check for number of images
for set in ['train', 'val', 'test']:
    n_normal = len(os.listdir(input_path + set + '/NORMAL'))
    n_infect = len(os.listdir(input_path + set + '/PNEUMONIA'))
    print('Set: {}, normal images: {}, pneumonia images: {}'.format(set, n_normal, n_infect))

# iterate over a dir
train_images = [TRAIN_DIR + 'NORMAL/' + i for i in os.listdir(TRAIN_DIR + 'NORMAL')] + [TRAIN_DIR + 'PNEUMONIA/' + i for i in os.listdir(TRAIN_DIR + 'PNEUMONIA')]
test_images = [TEST_DIR + 'NORMAL/' + i for i in os.listdir(TEST_DIR + 'NORMAL')] + [TEST_DIR + 'PNEUMONIA/' + i for i in os.listdir(TEST_DIR + 'PNEUMONIA')]
val_images = [VAL_DIR + 'NORMAL/' + i for i in os.listdir(VAL_DIR + 'NORMAL')] + [VAL_DIR + 'PNEUMONIA/' + i for i in os.listdir(VAL_DIR + 'PNEUMONIA')]
# print("Train images: ",len(train_images))
# print("Test images: ",len(test_images))
# print("Val images: ",len(val_images))

def read_and_prep_data(images):
  data = []
  for img in images:
      imag = cv2.imread(img, cv2.IMREAD_COLOR)
      image = cv2.resize(imag, (ROWS,COLS), interpolation=cv2.INTER_CUBIC)
      data.append(np.array(image))
  return data

# make labels
train_labels = []
for img in os.listdir(TRAIN_DIR + 'NORMAL'):
  train_labels.append('normal')
for img in os.listdir(TRAIN_DIR + 'PNEUMONIA'):
  train_labels.append('pneumonia')

test_labels = []
for img in os.listdir(TEST_DIR + 'NORMAL'):
  test_labels.append('normal')
for img in os.listdir(TEST_DIR + 'PNEUMONIA'):
  test_labels.append('pneumonia')

val_labels = []
for img in os.listdir(VAL_DIR + 'NORMAL'):
  val_labels.append('normal')
for img in os.listdir(VAL_DIR + 'PNEUMONIA'):
  val_labels.append('pneumonia')

#print(train_labels)

# convert to numpy array and read_and_prep_data
train = np.array(read_and_prep_data(train_images))
test = np.array(read_and_prep_data(test_images))
val = np.array(read_and_prep_data(val_images))
#np.expand_dims(  , axis=-1)
print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))
print("Val shape: {}".format(val.shape))

# normalization
train = train/255.0
test = test/255.0
val = val/255.0

# show train images
# fig, ax = plt.subplots(5, 2)
# fig.set_size_inches(15, 15)
# for i in range(3):
#     for j in range(2):
#         l = rn.randint(0, len(train_labels))
#         ax[i, j].imshow(train[l])
#         ax[i, j].set_title('Xray: ' + train_labels[l])
#
# plt.tight_layout()
# plt.show()


# LabelEncoder
le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
train_labels_enc = to_categorical(train_labels_enc,2)

test_labels_enc = le.fit_transform(test_labels)
test_labels_enc = to_categorical(test_labels_enc,2)

val_labels_enc = le.fit_transform(val_labels)
val_labels_enc = to_categorical(val_labels_enc,2)


base_model = VGG16(include_top=False, weights='imagenet', input_shape=(ROWS,COLS,CHANNELS), pooling='avg')

# summary of the architecture of the chosen model
base_model.summary()

# adding fully connected layers
model = Sequential()
model.add(base_model)

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))

# data augmentation
aug = ImageDataGenerator(fill_mode="constant",rotation_range=15, zoom_range=0.1,
	width_shift_range=0.1, height_shift_range=0.15, shear_range=0.1)

aug.fit(train)

epochs=20
batch_size = 64

model.summary()

base_model.trainable = False

#unfreze last block for fine tuning
# for i in range(len(base_model.layers)):
#     print(i, base_model.layers[i])
# for layer in base_model.layers[15:]:
#     layer.trainable = True
# for layer in base_model.layers[0:15]:
#     layer.trainable = False


# for layer in base_model.layers:
#     layer.trainable=True

optimizer = Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Train shape: ',train.shape)


#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#, callbacks=[early_stopping]
#, save_to_dir='images', save_prefix='aug', save_format='png'
History = model.fit_generator(aug.flow(train, train_labels_enc, batch_size=batch_size), epochs=epochs, validation_data = (val, val_labels_enc), verbose=1, steps_per_epoch=train.shape[0] // batch_size)
#model.save('vgg16_(feature_extractor)_adam_categorical.h5')
#model.save_weights('vgg16_denselayers.h5')

scores = model.evaluate(test, test_labels_enc)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = model.predict(test)
predictions_classes = predictions.argmax(axis=1)
#print('normal-0, pneumonia-1')
#print('True labels: ', test_labels_enc)
#print('Predicted classes', le.inverse_transform(predictions_classes))
#print("Predicted class on first image:",predictions_classes[0])
print('First prediction on first image:', predictions[0])

# confusion
predict = np.round(model.predict(test),0)
#print('rounded test\n',predict)
target = ['normal','pneumonia']
classification_metrics = classification_report(test_labels_enc, predict, target_names=target)
print(classification_metrics)

# graphs
# Accuracy
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
# Loss
plt.plot(History.history['val_loss'])
plt.plot(History.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation set', 'Training set'], loc='upper left')
plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
     plt.imshow(cm, interpolation='nearest', cmap=cmap)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45)
     plt.yticks(tick_marks, classes)

     thresh = cm.max() / 2.
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

     plt.tight_layout()
     plt.ylabel('True label')
     plt.xlabel('Predicted label')
     plt.show()


test_true = np.argmax(test_labels_enc, axis=1)
conf_mtx = confusion_matrix(test_true, predictions_classes)
plot_confusion_matrix(conf_mtx, classes=['normal', 'pneumonia'])

# Look at confusion matrix
accuracy = accuracy_score(test_labels, le.inverse_transform(predictions_classes))
conf_mat = confusion_matrix(test_labels, le.inverse_transform(predictions_classes))
true_negative, false_postive, false_negative, true_posiitve = conf_mat.ravel()

print('\n','-'*20,' TEST METRICS', '-'*20)
precision = true_posiitve / (true_posiitve + false_postive) * 100
recall = true_posiitve / (true_posiitve + false_negative) * 100
print('\tAccuracy: {}%'.format(accuracy))
print('\tPrecision: {}%'.format(precision))
print('\tRecall: {}%'.format(recall))
print('\tF1-score: {}'.format(2*precision*recall/(precision+recall)))


#gradcam
paths = ['test_normal_2.jpeg', 'test_normal.jpeg', 'test_pneumonia_virus.jpeg', 'test_pneumonia_bacteria.jpeg']

for path in paths:
    path = os.path.join("image", path)
    orig_img = np.array(load_img(path, target_size=(224, 224)), dtype=np.uint8)
    img = np.array(load_img(path, target_size=(224, 224)), dtype=np.float64)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img)
    predictions_classes = predictions.argmax(axis=1)

    gradcam = gradcamutils.grad_cam(base_model, img, layer_name='block5_conv3')
    gradcamplus = gradcamutils.grad_cam_plus(base_model, img, layer_name='block5_conv3')
    print(path)
    print("class activation map for:", predictions, le.inverse_transform(predictions_classes))
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(131)
    plt.imshow(orig_img)
    plt.title("input image")
    plt.subplot(132)
    plt.imshow(orig_img)
    plt.imshow(gradcam, alpha=0.8, cmap="jet")
    plt.title("Grad-CAM")
    plt.subplot(133)
    plt.imshow(orig_img)
    plt.imshow(gradcamplus, alpha=0.8, cmap="jet")
    plt.title("Grad-CAM++")
    plt.show()
