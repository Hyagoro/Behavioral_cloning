import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random

samples = []

first = True
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not first:
            samples.append(line)
        first = False

train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)
correction = 0.2

def bright_trans(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 2] = img[:, :, 2] * (0.2 + np.random.uniform())
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def generator(samples, batch_size=32, validation=False):
    num_samples = len(samples)
    while 1:
        # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                rnd = random.randrange(0, 3, 1)
                name = 'data/IMG/' + batch_sample[rnd].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                if rnd == 1:
                    center_angle += correction
                elif rnd == 2:
                    center_angle -= correction

                crop_img_top_ = center_image[65:160, 0:320, :]  # Crop from x, y, w, h -> 100, 200, 300, 400
                crop_img_bot_ = crop_img_top_[0:75, 0:320, :]
                resized_image = cv2.resize(crop_img_bot_, (200, 66), interpolation = cv2.INTER_AREA)
                blur = cv2.GaussianBlur(resized_image, (5, 5), 0)

                if not validation:
                    blur = bright_trans(blur)

                images.append(blur / 255.0 - 0.5)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32, validation=True)

# channel, row, column = 3, 90, 320  # Trimmed image format

kernel1_s = (1, 1)
kernel3_s = (3, 3)
kernel5_s = (5, 5)
kernel7_s = (7, 7)
pool_2s = (2, 2)
pool_4s = (4, 4)
# Keras model construction part
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
# model.add(Cropping2D(cropping=((65, 25), (0, 0))))
# model.add(MaxPooling2D(pool_size=pool_s))
# model.add(BatchNormalization())
model.add(Conv2D(24, kernel5_s, strides=(2, 2), input_shape=(66, 200, 3), activation='relu'))#strides=(2, 2),
model.add(Conv2D(36, kernel5_s, strides=(2, 2), activation='relu'))#strides=(2, 2),
model.add(Conv2D(48, kernel5_s, strides=(2, 2), activation='relu'))#strides=(2, 2),
model.add(Conv2D(64, kernel3_s, activation='relu'))
model.add(Conv2D(64, kernel3_s, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))#100
model.add(Dense(50))#50
model.add(Dense(10))#10
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()
adam = Adam(lr=0.01)
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')

print("Metrics")
# print the keys contained in the history object
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

# Can't run this on the server
# plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
