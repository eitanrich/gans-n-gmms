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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import image_batch_provider
from utils import mfa_sgd_training
import mfa_utils
import mfa
import ndb

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../Datasets/CelebA/img_align_celeba')
    parser.add_argument('--output_dir', help='Parent directory for storing all trained models', default='./restuls')
    parser.add_argument('--ndb_dir', help='NDB cache directory', default='./restuls/celeba_ndb_cache')
    args = parser.parse_args()

    model_name = 'celeba_mfa_200c_10l'
    model_dir = os.path.join(args.output_dir, model_name)

    image_shape = (64, 64)
    num_train = 80000
    num_test = 20000

    # Load the pre-trained model (run mfa_train_celeba.py first to train)
    gmm_model = mfa.MFA()
    gmm_model.load(os.path.join(model_dir, 'final_flat_model'))

    print('Loaded MFA model with {} components, data and and latent dimensions of {}'.format(
        len(gmm_model.components), gmm_model.components[0]['A'].shape))

    # First generate some random mosaics - just for fun
    print('Generating mosaic images...')
    mosaic_dir = os.path.join(model_dir, 'final_flat_mosaic')
    os.makedirs(mosaic_dir, exist_ok=True)
    for i in range(10):
        samples = gmm_model.draw_samples(16*9, add_noise=False)
        images = mfa_utils.to_images(samples, image_shape[0], image_shape[1])
        scipy.misc.imsave(os.path.join(mosaic_dir, '{}.jpeg'.format(i)), mfa_utils.images_to_mosaic(images))

    # Now generate images for evaluation
    print('Generating {} random images for evaluation...'.format(num_test))
    samples = gmm_model.draw_samples(num_test, add_noise=False)
    output_dir = os.path.join(model_dir, 'final_root_generated')
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_test):
        image = mfa_utils.to_image_8u(samples[i], image_shape[0], image_shape[1])
        imageio.imwrite(os.path.join(output_dir, '{}.png'.format(i)), image)

    # Perform NDB evaluation of the trained model
    image_provider = image_batch_provider.ImageBatchProvider(args.dataset_dir,
                                                             output_size=image_shape,
                                                             crop_bbox=(25, 50, 128, 128),
                                                             flatten=True,
                                                             list_file=os.path.join(args.dataset_dir, '../list_eval_partition.txt'))

    print('Reading train samples')
    train_samples = image_provider.get_random_samples(num_train)
    os.makedirs(args.ndb_dir, exist_ok=True)

    for num_bins in(100, 200, 300):
        print('Performing NDB evaluation for K={}'.format(num_bins))

        # Initializng NDB
        celeba_ndb = ndb.NDB(training_data=train_samples, number_of_bins=num_bins, whitening=True, cache_folder=args.ndb_dir)

        # Evaluating MFA samples
        images_folder = os.path.join(model_dir, 'final_flat_generated')
        mfa_provider = image_batch_provider.ImageBatchProvider(images_folder, flatten=True, test_set_ratio=0)
        celeba_ndb.evaluate(mfa_provider.get_random_samples(num_test), model_label=model_name)


if __name__ == "__main__":
    main(sys.argv)

