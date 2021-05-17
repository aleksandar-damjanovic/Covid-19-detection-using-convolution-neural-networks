import numpy as np # linear algebra
import pandas as pd

import os
for dirname, _, filenames in os.walk('C:/Users/coa/Downloads/covid19/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import random
random.seed(123)

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import cv2
import datetime
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from PIL import Image
import keras
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import VGG19

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import auc,roc_curve,roc_auc_score

optimizer = keras.optimizers.Adam(learning_rate = 1e-5)

#define the callbacks
early_stopping = [EarlyStopping(monitor = 'val_loss', patience = 10),
                 ModelCheckpoint(filepath = 'best_model.h5', monitor = 'val_loss', save_best_only = True)]


base_dir = 'C:/Users/coa/Downloads/covid19/'
train_dir = 'C:/Users/coa/Downloads/covid19/train/'
test_dir = "C:/Users/coa/Downloads/covid19/test/"
val_dir = "C:/Users/coa/Downloads/covid19/val/"

# train dataset
train_normal = train_dir + 'NORMAL/'
train_cov = train_dir + 'COVID/'

# test dataset
test_normal = test_dir + 'NORMAL/'
test_cov = test_dir + 'COVID/'

# validation dataset
val_normal = val_dir + 'NORMAL/'
val_cov = val_dir + 'COVID/'

print('Train Normal:', len(os.listdir(train_normal)))
print('Test Normal:', len(os.listdir(test_normal)))
print('Val Normal:', len(os.listdir(val_normal)))
print('\n')

#covid condition
print('Train Covid:', len(os.listdir(train_cov)))
print('Test Covid:', len(os.listdir(test_cov)))
print('Val Covid:', len(os.listdir(val_cov)))


normal = Image.open(train_normal + os.listdir(train_normal)[0])
cov = Image.open(train_cov + os.listdir(train_cov)[0])

fig = plt.figure(figsize = (10, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Normal X-ray', fontsize = 15)
plt.imshow(normal, cmap = 'gray')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Covid X-ray', fontsize = 15)
plt.imshow(cov, cmap = 'gray')

plt.savefig('X-ray.png')
plt.show()

plt.figure(figsize = (10, 6))
sns.barplot(x = ['NORMAL', 'COVID'],
            y = [len(os.listdir(train_normal)), len(os.listdir(train_cov))],
            palette = 'Spectral')
plt.xlabel('Condition', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.title('NORMAL and COVID Condition in Training Set', fontsize = 15)
plt.show()


# augment train and validation dataset to prevent overfitting by increasing number of images
train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   # randomly rotate images
                                   rotation_range=40,

                                   # randomly shear angles
                                   shear_range=0.2,

                                   # randomly zoom images
                                   zoom_range=0.2,

                                   # randomly shift images
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,

                                   # randomly flip images
                                   horizontal_flip=True,

                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=40,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

# rescale test dataset without augmentation since real world data is not augmented
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    # target directory
    train_dir,

    # resize to 180x180
    target_size=(180, 180),

    # size batches of data
    batch_size=16,

    # since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(180, 180),
                                                  batch_size=16,
                                                  class_mode='binary',
                                                  shuffle=False)

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(180, 180),
                                                batch_size=16,
                                                class_mode='binary')




def picture_separation(folder):
    '''divide the mixed pictures into NORMAL and COVID & add labels to these'''

    X = []
    y = []
    image_list = []

    for foldername in os.listdir(folder):
        if not foldername.startswith('.'):
            if foldername == "NORMAL":
                label = 0
            elif foldername == "COVID":
                label = 1
            else:
                label = 2

            for image_filename in os.listdir(folder + "/" + foldername):
                img_file = cv2.imread(folder + "/" + foldername + '/' + image_filename, 0)

                if img_file is not None:
                    img = cv2.resize(img_file, (64, 64))
                    img_arr = img_to_array(img) / 255
                    X.append(img_arr)
                    y.append(label)
                    image_list.append(foldername + '/' + image_filename)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, image_list

X_train, y_train, img_train = picture_separation(train_dir)

train_df = pd.DataFrame(img_train, columns = ["images"])
train_df["target"] = y_train

#preview
print(train_df.head())

X_val, y_val, img_val = picture_separation(val_dir)

val_df = pd.DataFrame(img_val, columns = ["images"])
val_df["target"] = y_val

#preview
print(val_df.head())

X_test, y_test, img_test = picture_separation(test_dir)

test_df = pd.DataFrame(img_test, columns = ["images"])
test_df["target"] = y_test

#preview
print(test_df.head())

full_data = pd.concat([train_df, test_df, val_df], axis = 0, ignore_index = True)
print(full_data.info())

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('\n')

print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('\n')

print('X_val shape:', X_val.shape)
print('y_val shape:', y_val.shape)

X_train = X_train.reshape(2800, 64*64).astype('float32')
X_test = X_test.reshape(400, 64*64).astype('float32')
X_val = X_val.reshape(52, 64*64).astype('float32')
#recheck
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_val shape:', X_val.shape)


early_stopping = [EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1),
                  ModelCheckpoint(filepath ='../../Desktop/cnn_model.h5', monitor ='val_loss', save_best_only = True)]

#initialize
cnn_model = models.Sequential()

#define model architecture
cnn_model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (180, 180, 3)))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(32, (3, 3), activation = 'relu'))
cnn_model.add(MaxPooling2D((2, 2)))

# fully connected layer
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation = 'relu'))
cnn_model.add(Dense(128, activation = 'relu'))

# one layer with sigmoid activation
cnn_model.add(Dense(1, activation = 'sigmoid'))

#compile
cnn_model.compile(loss = 'binary_crossentropy',
                  optimizer = keras.optimizers.Adam(0.0001),
                  metrics = ['acc'])

# get summary
cnn_model.summary()

# plot
# plot_model(cnn_model, to_file = 'cnn_model_plot.png', show_shapes = True, show_layer_names = True)

start = datetime.datetime.now()

# fit
cnn_history = cnn_model.fit(train_generator,
                            steps_per_epoch=2800 // 16,
                            epochs=30,
                            validation_data=test_generator,
                            validation_steps=400 // 16,
                            callbacks=early_stopping)

# time
end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))

# save model
cnn_model.save('CNN_model.h5')

fig , ax = plt.subplots(1, 2)
fig.set_size_inches(20, 8)

cnn_train_acc = cnn_history.history['acc']
cnn_train_loss = cnn_history.history['loss']
cnn_val_acc = cnn_history.history['val_acc']
cnn_val_loss = cnn_history.history['val_loss']

epochs = range(1, len(cnn_train_acc) + 1)

ax[0].plot(epochs, cnn_train_acc, 'go-' , label='Training Accuracy')
ax[0].plot(epochs, cnn_val_acc, 'yo-' , label='Validation Accuracy')
ax[0].set_title('CNN Model Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, cnn_train_loss, 'go-', label='Training Loss')
ax[1].plot(epochs, cnn_val_loss, 'yo-', label='Validation Loss')
ax[1].set_title('CNN Model Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")

plt.show()

# evaluate
print('Train loss & accuracy:', cnn_model.evaluate(train_generator))
print('\n')
print('Test loss & accuracy:', cnn_model.evaluate(test_generator))

# define target for testing
y_test = test_generator.classes

# make prediction
yhat_test = cnn_model.predict(test_generator) > 0.5

# get confusion matrix
cm = confusion_matrix(y_test, yhat_test)
print(cm)

# visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# get classification report
print('Model: CNN', '\n', classification_report(y_test, yhat_test, target_names=['NORMAL (Class 0)', 'COVID (Class 1)']))

fpr, tpr, thresholds = roc_curve(y_test, yhat_test)
auc = roc_auc_score(y_test, yhat_test)
print('AUC:', auc)

plt.figure(figsize = (10, 5))
plt.plot(fpr, tpr, color='blue', label = 'ROC curve (area = %f)' % auc)
plt.plot([0, 1], [0, 1], linestyle='--', color='darkgreen')
plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CNN Model X-ray Covid Classification')
plt.legend(loc="lower right")
plt.show()
