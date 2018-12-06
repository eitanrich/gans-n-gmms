import os
import sys
import numpy as np
import scipy.misc
import imageio
import cv2
import argparse
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import image_batch_provider
from utils import mfa_sgd_training
import mfa_utils
import mfa
import ndb

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../Datasets/MNIST')
    parser.add_argument('--output_dir', help='Parent directory for storing all trained models', default='./restuls')
    parser.add_argument('--ndb_dir', help='NDB cache directory', default='./restuls/mnist_ndb_cache')
    args = parser.parse_args()

    model_name = 'mnist_mfa_150c_5l'
    model_dir = os.path.join(args.output_dir, model_name)

    image_shape = (28, 28, 1)
    num_train = 50000
    num_test = 10000

    # Load the pre-trained model (run mfa_train_mnist.py first to train)
    gmm_model = mfa.MFA()
    gmm_model.load(os.path.join(model_dir, 'final_gmm'))

    print('Loaded MFA model with {} components, data and and latent dimensions of {}'.format(
        len(gmm_model.components), gmm_model.components[0]['A'].shape))

    # First generate some random mosaics - just for fun
    print('Generating mosaic images...')
    mosaic_dir = os.path.join(model_dir, 'final_mosaic')
    os.makedirs(mosaic_dir, exist_ok=True)
    for i in range(10):
        samples = gmm_model.draw_samples(16*9, add_noise=False)
        images = mfa_utils.to_images(samples, *image_shape)
        scipy.misc.imsave(os.path.join(mosaic_dir, '{}.jpeg'.format(i)), mfa_utils.images_to_mosaic(images))

    # Now generate images for evaluation
    print('Generating {} random images for evaluation...'.format(num_test))
    samples = gmm_model.draw_samples(num_test, add_noise=False)
    samples = np.maximum(0.0, np.minimum(1.0, samples))
    output_dir = os.path.join(model_dir, 'old_generated')
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_test):
        image = mfa_utils.to_image_8u(samples[i], *image_shape)
        imageio.imwrite(os.path.join(output_dir, '{}.png'.format(i)), image)

    # Perform NDB evaluation of the trained model
    image_provider = image_batch_provider.ImageBatchProvider(args.dataset_dir,
                                                             flatten=True,
                                                             batch_size=512,
                                                             mirror=False)

    print('Reading train samples')
    train_samples = image_provider.get_random_samples(num_train)
    os.makedirs(args.ndb_dir, exist_ok=True)

    images_folder = os.path.join(model_dir, 'old_generated')
    mfa_provider = image_batch_provider.ImageBatchProvider(images_folder, flatten=True, mirror=False,
                                                           test_set_ratio=0, read_as_gray=True)

    for num_bins in(50, 100, 200):
        print('Performing NDB evaluation for K={}'.format(num_bins))

        # Initializng NDB
        mnist_ndb = ndb.NDB(training_data=train_samples, number_of_bins=num_bins, whitening=False, cache_folder=args.ndb_dir)

        # Evaluating MFA samples
        mnist_ndb.evaluate(mfa_provider.get_random_samples(num_test), model_label=model_name)


if __name__ == "__main__":
    main(sys.argv)
