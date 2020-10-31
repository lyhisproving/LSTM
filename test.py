import tensorflow as tf
import numpy as np


def one_hot(label, num_class):
    label = np.array(label)
    num_label = label.shape[0]
    index = np.arange(num_label) * num_class
    out = np.zeros((num_label, num_class))
    out.flat[index + label.ravel()] = 1
    return out


x_train_in, x_train_out, x_test_in, x_test_out, y_train, y_test = load_data(
    'data/TrainSamples_10_26.txt', 'data/TestSamples_10_26.txt',
    'data/TrainLabel_10_26.txt', 'data/TestLabel_10_26.txt')

batch_size = 64
seq_len_in = x_train_in.shape[1]
seq_len_out = x_train_out.shape[1]
embed_size = 1
learning_rate = 0.0001
hidden_size = 128
iterations = 3000
n_classes = 2
