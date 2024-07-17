from keras.callbacks import Callback

# Define a callback to track loss
class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
