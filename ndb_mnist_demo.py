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


def sample_from(samples, number_to_use):
    assert samples.shape[0] >= number_to_use
    rand_order = np.random.permutation(samples.shape[0])
    return samples[rand_order[:number_to_use], :]


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', help='folder containing the MNIST dataset', default='./data/mnist')
    parser.add_argument('--num_bins', default=50)
    return parser.parse_args()


def main(argv):
    args = parse_args(argv)
    train_val_samples, _, test_samples, test_labels = load_mnist_data(args.dataset_folder)
    n_test = 7500
    n_val = n_test
    n_train = train_val_samples.shape[0] - n_val

    print('Splitting MNIST training data to random train/val ({}/{}).'.format(n_train, n_val))
    rand_order = np.random.permutation(train_val_samples.shape[0])
    train_samples = train_val_samples[rand_order[:n_train]]
    val_samples = train_val_samples[rand_order[n_train:]]

    print('Initialize NDB bins with training samples')
    mnist_ndb = NDB(training_data=train_samples, number_of_bins=args.num_bins, whitening=False)
                    # bins_file='mnist_{}.pkl'.format(args.num_bins))

    print('Evaluating {} validation samples (randomly split from the train samples - should be very similar)'.format(n_val))
    mnist_ndb.evaluate(sample_from(val_samples, n_val), 'Validation')

    print('Evaluating {} test samples (different writers - can be somewhat different)'.format(n_test))
    mnist_ndb.evaluate(sample_from(test_samples, n_test), 'Test')

    print('Simulate a deviation from the data distribution (mode collapse) by sampling from specific labels (digits)')
    mnist_ndb.evaluate(sample_from(test_samples[test_labels < 9, :], n_test), 'Test0-8')
    mnist_ndb.evaluate(sample_from(test_samples[test_labels < 8, :], n_test), 'Test0-7')

    mnist_ndb.plot_results()

if __name__ == "__main__":
    main(sys.argv)
