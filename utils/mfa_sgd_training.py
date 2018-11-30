import tensorflow as tf
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils import mfa
from utils import mfa_utils
from utils import mfa_tf
from utils import image_batch_provider


def train(num_components, latent_dimension, init_method='km',
          image_provider=None, batch_size=200, test_size=0, test_set=None,
          learning_rate=1e-4, max_iters=10000, init_gmm=None,
          init_whiten=True, training_whiten=False,
          image_shape=None, out_folder=None):
    """
    Run SGD training of a Mixture of Factor Analyzers (MFA) model
    :param num_components: Number of mixture components (K)
    :param latent_dimension: The latent model dimension (dimension of z)
    :param init_method: The initialization method: 'km' (K-means) / 'rnd' (random samples)
    :param image_provider: A data-provider object
    :param batch_size: Training batch size
    :param test_size: Size of the test set to use
    :param test_set: Optional - the actual test samples
    :param learning_rate: The learning rate
    :param max_iters: Number of iterations
    :param init_gmm: Optional - the GMM model to start from
    :param init_whiten: Whiten data during initialization
    :param training_whiten: Whiten data during training
    :param image_shape: Image shape tuple (w, h) or (w, h, ch) for display purposes
    :param out_folder: Folder to write model and debug images to
    :return: The resulting GMM model
    """
    if test_set is not None:
        test_size = test_set.shape[0]

    # Reset everything so that variables will not accumulate from previous run
    tf.reset_default_graph()

    if not init_gmm:
        print('Initial guess...')
        if init_whiten:
            _, init_std = mfa_utils.get_dataset_mean_and_std(image_provider)
        else:
            init_std = 1.0
        init_samples_per_comp = 300
        init_n = min(image_provider.num_train_images, max(num_components * init_samples_per_comp, 10000))
        print('Collecting an initial sample of', init_n, 'samples...')
        init_samples = image_provider.get_random_samples(init_n)
        init_gmm = mfa_utils.gmm_initial_guess(init_samples, num_components, latent_dimension, clustering_method=init_method,
                                               component_model='fa', dataset_std=init_std, default_noise_std=0.15)

    if training_whiten:
        training_dataset_mean, training_dataset_std = mfa_utils.get_dataset_mean_and_std(image_provider)
    else:
        training_dataset_mean, training_dataset_std = (0.0, 1.0)

    d, l = init_gmm.components[0]['A'].shape
    assert len(init_gmm.components) == num_components and l == latent_dimension

    # Raw TF parameters
    G_PI, G_MU, G_A, G_D = mfa_tf.init_raw_parms_from_gmm(init_gmm)
    theta_G = [G_PI, G_MU, G_A, G_D]

    X = tf.placeholder(tf.float32, shape=[None, d])

    G_loss = -1.0 * mfa_tf.get_log_likelihood(X, *theta_G)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)
    # LL_values = mfa_tf.get_per_sample_per_component_log_prob(X, *theta_G)

    if out_folder:
        init_gmm.save(os.path.join(out_folder, 'init_gmm'))
        saver = tf.train.Saver(var_list=theta_G)
        log_file = open(os.path.join(out_folder, 'log.csv'), 'w')
        out_images_folder = os.path.join(out_folder, 'images')
        os.makedirs(out_images_folder, exist_ok=True)

    def get_training_batches():
        while True:
            samples = image_provider.get_next_minibatch_samples()
            samples = (samples - training_dataset_mean) / training_dataset_std
            yield samples
    training_batch_generator = get_training_batches()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if test_size > 0 and test_set is None:
        test_set = image_provider.get_test_samples(test_size)

    print('Starting Mixture of Factor Analyzers ML SGD Training ({} iterations)...'.format(max_iters))
    test_step = 100
    all_test_ll = {}
    all_train_ll = {}
    for it in range(max_iters):

        mb_samples = next(training_batch_generator)

        # Visualize and save current model
        if (it % (test_step*5) == 0) or it == (max_iters-1):
            if out_folder:
                print('[saving model]', end='', flush=True)
                saver.save(sess, os.path.join(out_folder, 'model', 'gmm_model'))
            print('[visualizing]', end='', flush=True)
            c_PI, c_MU, c_A, c_D = sess.run([G_PI, G_MU, G_A, G_D])
            est_gmm = mfa_tf.raw_to_gmm(c_PI, c_MU, c_A, c_D)
            mfa_utils.visualize_trained_model(est_gmm, it, out_folder=out_images_folder, image_shape=image_shape,
                                              mean=training_dataset_mean, std=training_dataset_std)

        print('.', end='', flush=True)

        # Calculate test-set log-likelihood
        if it % test_step == 0 and test_size > 0:
            test_ll = 0
            for test_idx in range(0, test_size, batch_size):
                print('x', end='', flush=True)
                c_loss, c_D = sess.run([G_loss, G_D],
                                       feed_dict={X: test_set[test_idx:min(test_idx+batch_size, test_size)]})
                test_ll -= c_loss
            test_ll /= test_size

            if it == 0:
                train_ll = -1.0/batch_size * sess.run(G_loss, feed_dict={X: mb_samples})
            else:
                train_ll = -1.0/batch_size * curr_loss

            mean_noise_variance = np.mean(np.power(c_D, 2.0))
            print('\nIteration {}: LL = {}, Test LL = {}, Mean noise std = {}'.format(it, train_ll, test_ll, mean_noise_variance))

            all_train_ll[it] = train_ll
            all_test_ll[it] = test_ll
            # Plot training progress
            plt.figure(4)
            plt.clf()
            iters = sorted(list(all_test_ll.keys()))
            plt.plot(iters, [all_train_ll[it] for it in iters], '-.', label='Train')
            plt.plot(iters, [all_test_ll[it] for it in iters], '-*', label='Test')
            plt.grid(True)
            plt.legend()
            plt.title('MFA Log-Likelihood')
            plt.pause(0.1)
            if out_folder:
                log_file.write('{},{},{},{}\n'.format(it, train_ll, test_ll, mean_noise_variance))
                log_file.flush()

        # Run an SGD iteration
        _, curr_loss = sess.run([G_solver, G_loss], feed_dict={X: mb_samples})

    print('MFA Training ended.')
    est_gmm.save(os.path.join(out_folder, 'final_gmm'))
    log_file.close()
    plt.figure(4)
    plt.savefig(os.path.join(out_folder, 'train-test-ll-graph.pdf'))
    return est_gmm
