import sys
import os
import argparse
from ndb import *
import numpy as np
import struct
from matplotlib import pyplot as plt
# from utils import image_batch_provider


def load_images(ubyte_file):
    with open(ubyte_file, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)
    return images.reshape([-1, 28*28]).astype(float)/255.0

def load_labels(ubyte_file):
    with open(ubyte_file, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
    return labels


def load_mnist_data(dataset_folder):
    print('Loading MNIST data...')
    train_samples = load_images(os.path.join(dataset_folder, 'train-images.idx3-ubyte'))
    train_labels = load_labels(os.path.join(dataset_folder, 'train-labels.idx1-ubyte'))
    test_samples = load_images(os.path.join(dataset_folder, 't10k-images.idx3-ubyte'))
    test_labels = load_labels(os.path.join(dataset_folder, 't10k-labels.idx1-ubyte'))
    return train_samples, train_labels, test_samples, test_labels

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', help='folder containing the MNIST dataset', default='./data/mnist')
    parser.add_argument('--num_bins', default=100)
    return parser.parse_args()


def main(argv):
    args = parse_args(argv)
    train_val_samples, _, test_samples, test_labels = load_mnist_data(args.dataset_folder)
    n_val = train_val_samples.shape[0]//6
    n_train = train_val_samples.shape[0] - n_val
    n_test = test_samples.shape[0]
    print('Dividing MNIST data to {}/{} (train/val) and {} (test)'.format(n_train, n_val, n_test))
    # Evaluate a small random batch from the training set - should be very similar to the reference bins
    rand_order = np.random.permutation(train_val_samples.shape[0])
    train_samples = train_val_samples[rand_order[:n_train]]
    val_samples = train_val_samples[rand_order[n_train:]]

    # Initialize the NDB bins with the training samples
    mnist_ndb = NDB(training_data=train_samples, number_of_bins=args.num_bins, whitening=False,
                    bins_file='mnist_{}.pkl'.format(args.num_bins))

    # Evaluate the validation samples (randomly split from the train samples - should be very similar)
    mnist_ndb.evaluate(val_samples, 'Validation')

    # Evaluate the Test set (different writers - can be somewhat different)
    mnist_ndb.evaluate(test_samples, 'Test')

    # Simulate a deviation from the data distribution (mode collapse) by sampling from specific labels (digits)
    mnist_ndb.evaluate(test_samples[test_labels < 9, :], 'Test0-8')
    mnist_ndb.evaluate(test_samples[test_labels < 8, :], 'Test0-7')

    mnist_ndb.plot_results()

if __name__ == "__main__":
    main(sys.argv)
