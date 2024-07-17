from keras.optimizers import Adam
from resnet50 import ResNet50
from learing_rate_schedule import lr_schedule
from preprocessing import input_shape
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import Callback, ReduceLROnPlateau
import numpy as np

model = ResNet50(input_shape=input_shape, classes=10)

model.compile(loss = 'categorical_crossentropy',
             optimizer = Adam(learning_rate = lr_schedule(0)),
             metrics = ['accuracy'])

model.summary()

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_ResNet50_model_{epoch:03d}_weights.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                            save_weights_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

