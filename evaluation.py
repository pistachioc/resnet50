from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
from preprocessing import x_test, y_test
from train import model
from model import filepath

# Evaluate model and generate F1-score report
correct = 0
total = 0

all_labels = []
all_predictions = []

# Assuming model is your Keras model and x_test, y_test are your test data
with tf.device('/GPU:0'):  # Specify GPU if available
    predictions = model.predict(x_test)  # Predict using the test data

    all_predictions = np.argmax(predictions, axis=1)
    all_labels = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded

accuracy = np.sum(all_predictions == all_labels) / len(all_labels) * 100

# Generate F1-score report
report = classification_report(all_labels, all_predictions,
                               target_names=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                             'truck'])

print('Accuracy on test set: {:.2f}%'.format(accuracy))
print('F1-Score Report:\n', report)

model.save(filepath=filepath)
