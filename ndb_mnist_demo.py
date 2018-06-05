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


def visualize_bins(bin_centers, is_different):
    k = bin_centers.shape[0]
    n_cols = 10
    n_rows = (k+n_cols-1)//n_cols
    for i in range(k):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(bin_centers[i, :].reshape([28, 28]))
        if is_different[i]:
            plt.plot([0, 27], [0, 27], 'r', linewidth=2)
        plt.axis('off')

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', help='folder containing the MNIST dataset', default='./data/mnist')
    parser.add_argument('--num_bins', default=100)
    return parser.parse_args()


def main(argv):
    args = parse_args(argv)
    train_val_samples, train_val_labels, test_samples, _ = load_mnist_data(args.dataset_folder)
    n_query = 10000
    n_val = 15000
    n_train = train_val_samples.shape[0] - n_val

    print('Splitting MNIST training data to random train/val ({}/{}).'.format(n_train, n_val))
    rand_order = np.random.permutation(train_val_samples.shape[0])
    train_samples = train_val_samples[rand_order[:n_train]]
    val_samples = train_val_samples[rand_order[n_train:]]
    val_labels = train_val_labels[rand_order[n_train:]]

    print('Initialize NDB bins with training samples')
    mnist_ndb = NDB(training_data=train_samples, number_of_bins=args.num_bins, significance_level=0.01, whitening=False,
                    bins_file='mnist_{}.pkl'.format(args.num_bins))

    print('Evaluating {} validation samples (randomly split from the train samples - should be very similar)'.format(n_query))
    mnist_ndb.evaluate(sample_from(val_samples, n_query), 'Validation')


    print('Simulate a deviation from the data distribution (mode collapse) by sampling from specific labels (digits)')
    biased_samples = sample_from(val_samples[val_labels < 9, :], n_query)
    results = mnist_ndb.evaluate(biased_samples, 'Val0-8')

    # Visualize the missing bins
    visualize_bins(mnist_ndb.bin_centers, results['Different-Bins'])
    plt.savefig('bins_with_Val0-8_results_{}.png'.format(args.num_bins))

    print('Evaluating {} test-set samples (different writers - can be somewhat different)'.format(n_query))
    mnist_ndb.evaluate(sample_from(test_samples, n_query), 'Test')

    # mnist_ndb.evaluate(sample_from(test_samples[test_labels < 8, :], n_test), 'Test0-7')

    plt.figure()
    mnist_ndb.plot_results()

if __name__ == "__main__":
    main(sys.argv)
