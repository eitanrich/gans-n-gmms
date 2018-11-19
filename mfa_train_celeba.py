import os
import sys
import numpy as np
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
from hierarchic_mfa_utils import split_data_by_model_components, flatten_hierarchic_model


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../Datasets/CelebA/img_align_celeba')
    parser.add_argument('--output_dir', help='Parent directory for storing all trained models', default='./restuls')
    parser.add_argument('--num_components', help='Number of (root level) mixture components', default=200)
    parser.add_argument('--samples_per_sub_component', help='For hierarchic (two-level) training, target number of samples per final component', default=400)
    parser.add_argument('--latent_dimension', help='Dimension of input factors z', default=10)
    args = parser.parse_args()

    image_shape = (64, 64)
    batch_size = 256
    test_size = batch_size*10
    image_provider = image_batch_provider.ImageBatchProvider(args.dataset_dir,
                                                             output_size=image_shape,
                                                             crop_bbox=(25, 50, 128, 128),
                                                             flatten=True,
                                                             batch_size=batch_size,
                                                             list_file=os.path.join(args.dataset_dir, '../list_eval_partition.txt'))
    output_folder = os.path.join(args.output_dir, 'celeba_mfa_{}c_{}l'.format(args.num_components, args.latent_dimension))
    print('Running MFA Teaining. Output folder is', output_folder)
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.isfile(os.path.join(output_folder, 'final_gmm.pkl')):
        gmm_model = mfa_sgd_training.train(num_components=args.num_components, latent_dimension=args.latent_dimension,
                         out_folder=output_folder, image_shape=image_shape, init_method='km',
                         image_provider=image_provider, batch_size=batch_size, test_size=test_size,
                         learning_rate=5e-5, max_iters=10000)
    else:
        print('Loading pre-trained root model...')
        gmm_model = mfa.MFA()
        gmm_model.load(os.path.join(output_folder, 'final_gmm'))

    # Hierarchic training...
    if args.samples_per_sub_component > 0:
        print('Now splitting each root component to sub-components...')
        if not os.path.isdir(os.path.join(output_folder, 'component_lists')):
            split_data_by_model_components(gmm_model, output_folder, image_provider, image_shape, batch_size)

        for comp_num in range(args.num_components):
            list_file = os.path.join(output_folder, 'component_lists', 'comp_{}.txt'.format(comp_num))
            comp_image_provider = image_batch_provider.ImageBatchProvider(args.dataset_dir,
                                                                     output_size=image_shape,
                                                                     crop_bbox=(25, 50, 128, 128),
                                                                     flatten=True,
                                                                     batch_size=batch_size,
                                                                     mirror=False,
                                                                     list_file=list_file)

            comp_out_folder = os.path.join(output_folder, 'hierarchic_model', 'comp_{}'.format(comp_num))
            if os.path.isfile(os.path.join(comp_out_folder, 'final_gmm.pkl')):
                print('Skipping component {} - already learned.'.format(comp_num))
            else:
                os.makedirs(comp_out_folder, exist_ok=True)
                num_sub_comps = comp_image_provider.num_train_images // args.samples_per_sub_component
                if num_sub_comps < 2:
                    print('No sub-components for component number {}.'.format(comp_num))
                    comp_gmm = mfa.MFA({0: gmm_model.components[comp_num]})
                    comp_gmm.components[0]['pi'] = 1.0
                    comp_gmm.save(os.path.join(comp_out_folder, 'final_gmm'))
                else:
                    print('Training {} sub-components for root component {}...'.format(num_sub_comps, comp_num))
                    for tries in range(3):
                        try:
                            mfa_sgd_training.train(num_components=num_sub_comps, latent_dimension=args.latent_dimension,
                                     out_folder=comp_out_folder, image_shape=image_shape, init_method='km',
                                     image_provider=comp_image_provider, batch_size=batch_size, test_size=comp_image_provider.num_test_images,
                                     learning_rate=5e-5, max_iters=5000)
                        except:
                            print('An error occured.')
                        else:
                            break
                    else:
                        print('Training of component {} failed!!!'.format(comp_num))

        print('Creating the final flat model...')
        flatten_hierarchic_model(gmm_model, output_folder)

    print('Done')

if __name__ == "__main__":
    main(sys.argv)

