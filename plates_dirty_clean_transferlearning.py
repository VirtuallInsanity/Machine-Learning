import os
import pandas as pd
import numpy as np
import cv2

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, Activation, BatchNormalization, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet import ResNet50

TRAIN_DIR = 'plates/train/'
TEST_DIR = 'plates/test/'
VAL_DIR = 'plates/val/'
input_path = 'plates/'

ROWS = 224
COLS = 224
CHANNELS = 3

for set in ['train', 'val']:
    n_cleaned = len(os.listdir(input_path + set + '/cleaned'))
    n_dirty = len(os.listdir(input_path + set + '/dirty'))
    print('Set: {}, cleaned plates: {}, dirty plates: {}'.format(set, n_cleaned, n_dirty))
n_test = len(os.listdir(TEST_DIR))
print('Set: test, number of test images: {}'.format(n_test))

train_images = [TRAIN_DIR + 'cleaned/' + i for i in os.listdir(TRAIN_DIR + 'cleaned')] + [TRAIN_DIR + 'dirty/' + i for i in os.listdir(TRAIN_DIR + 'dirty')]
test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
val_images = [VAL_DIR + 'cleaned/' + i for i in os.listdir(VAL_DIR + 'cleaned')] + [VAL_DIR + 'dirty/' + i for i in os.listdir(VAL_DIR + 'dirty')]


def read_and_prep_data(images):
  data = []
  for img in images:
      imag = cv2.imread(img, cv2.IMREAD_COLOR)
      image = cv2.resize(imag, (ROWS,COLS), interpolation=cv2.INTER_CUBIC)
      data.append(np.array(image))
  return data

# make labels
train_labels = []
for img in os.listdir(TRAIN_DIR + 'cleaned'):
  train_labels.append(0)
for img in os.listdir(TRAIN_DIR + 'dirty'):
  train_labels.append(1)

val_labels = []
for img in os.listdir(VAL_DIR + 'cleaned'):
  val_labels.append(0)
for img in os.listdir(VAL_DIR + 'dirty'):
  val_labels.append(1)

print(train_labels)

# convert to numpy array and read_and_prep_data
train = np.array(read_and_prep_data(train_images))
test = np.array(read_and_prep_data(test_images))
val = np.array(read_and_prep_data(val_images))
print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))
print("Val shape: {}".format(val.shape))

# normalization
train = train/255.0
test = test/255.0
val = val/255.0

# import random as rn
# # show train images
# fig, ax = plt.subplots(5, 2)
# fig.set_size_inches(15, 15)
# for i in range(3):
#     for j in range(2):
#         l = rn.randint(0, len(train_labels))
#         ax[i, j].imshow(train[l])
#         ax[i, j].set_title(train_labels[l])
#
# plt.tight_layout()
# plt.show()

from skimage import exposure
def augfunc(image):
    aug = exposure.adjust_gamma(image, 1.5)
    return aug
#base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(ROWS,COLS,CHANNELS), pooling='avg')
#base_model = VGG16(include_top=False, weights='imagenet', input_shape=(ROWS,COLS,CHANNELS), pooling='avg')
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, CHANNELS))
model = Sequential()
model.add(base_model)

model.add(Flatten())
#model.add(GaussianNoise(0.1))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print("input shape ",model.input_shape)
print("output shape ",model.output_shape)



# data augmentation
aug = ImageDataGenerator(preprocessing_function = augfunc,rotation_range=15, zoom_range=0.1,
	width_shift_range=0.08, height_shift_range=0.08, shear_range=0.05, fill_mode="constant", horizontal_flip=True, vertical_flip = True)

aug.fit(train)

epochs=20
batch_size = 12

#base_model.summary()
#model.summary()

# use all layers
for layer in base_model.layers:
    layer.trainable=True

# unfreze last block for fine tuning
# for i in range(len(base_model.layers)):
#     print(i, base_model.layers[i])
#
# for layer in base_model.layers[11:]:
#     layer.trainable = True
# for layer in base_model.layers[0:11]:
#     layer.trainable = False

# freze all vgg16
#base_model.trainable = False


#optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
optimizer = Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print('Train shape: ',train.shape)

History = model.fit_generator(aug.flow(train, train_labels, batch_size=batch_size, save_to_dir='images', save_prefix='aug', save_format='png'), epochs=epochs, validation_data = (val, val_labels), verbose=1, steps_per_epoch=train.shape[0] // batch_size)
#
#model.save_weights('InceptionResNetV2_weights.h5')

predictions = model.predict(test)
print('cleaned-0, dirty-1')
i=0
for predict in predictions:
    if predict > 0.75:
        print('dirty', i, predict)
        i+=1
    else:
        print('cleaned', i, predict)
        i+=1

print('First prediction on first image:', predictions[0])


# make submission
sub_df = pd.read_csv('sample_submission.csv')

sub_df['label'] = predictions
sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

print(sub_df['label'].value_counts())
sub_df.to_csv('InceptionResNetV2.csv', index=False)


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
