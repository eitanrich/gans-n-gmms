import os
import sys
import numpy as np
import argparse
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from matplotlib import pyplot as plt
from utils import image_batch_provider
from utils import mfa_sgd_training
import mfa_utils
import mfa


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../Datasets/MNIST')
    parser.add_argument('--output_dir', help='Parent directory for storing all trained models', default='./restuls')
    parser.add_argument('--num_components', help='Number of (root level) mixture components', default=150)
    parser.add_argument('--latent_dimension', help='Dimension of input factors z', default=5)
    args = parser.parse_args()

    image_shape = (28, 28, 1)
    batch_size = 512
    image_provider = image_batch_provider.ImageBatchProvider(args.dataset_dir,
                                                             flatten=True,
                                                             batch_size=batch_size,
                                                             mirror=False)
    output_folder = os.path.join(args.output_dir, 'mnist_mfa_{}c_{}l'.format(args.num_components, args.latent_dimension))
    print('Running MFA Teaining. Output folder is', output_folder)
    os.makedirs(output_folder, exist_ok=True)

    mfa_sgd_training.train(num_components=args.num_components, latent_dimension=args.latent_dimension,
                                       out_folder=output_folder, image_shape=image_shape, init_method='km',
                                       init_whiten=False, image_provider=image_provider, batch_size=batch_size,
                                       test_size=batch_size*20, learning_rate=2e-5, max_iters=6000)

    print('Done')
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

