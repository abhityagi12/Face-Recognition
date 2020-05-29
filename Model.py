from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


IMAGE_SIZE=[224,224]

train_path='Datasets/Train'
test_path='Datasets/Test'

vgg=VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)

for layer in vgg.layers:
	layer.trainable=False

folders=glob('Datasets/Train/*')

x=Flatten()(vgg.output)

prediction=Dense(len(folders),activation='softmax')(x)

model=Model(inputs=vgg.input, outputs=prediction)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,rotation_range=70,brightness_range=[0.2,1.0])

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(train_path, target_size=(224,224), batch_size=32, class_mode='categorical')

test_set=test_datagen.flow_from_directory(test_path, target_size=(224,224), batch_size=32, class_mode='categorical')

r=model.fit_generator(training_set, validation_data=test_set, epochs=10, steps_per_epoch=len(training_set), validation_steps=len(test_set))


import tensorflow as tf
from keras.models import load_model

model.save('facefeatures_model2.h5')

plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['acc'], label="train acc")
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')