import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import mfa
import cv2
import time
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def to_image(sample, w, h, ch=3):
    if ch>1:
        return np.maximum(0.0, np.minimum(1.0, sample.reshape([h, w, ch])))
    return np.maximum(0.0, np.minimum(1.0, sample.reshape([h, w])))


def to_images(samples, w, h, ch=3):
    if ch>1:
        return np.maximum(0.0, np.minimum(1.0, samples.reshape([-1, h, w, ch])))
    return np.maximum(0.0, np.minimum(1.0, samples.reshape([-1, h, w])))


def images_to_mosaic(samples, rows=9, cols=16):
    if len(samples.shape) == 3:
        samples = np.expand_dims(samples, axis=3)
    n, w, h, ch = samples.shape
    mosaic = np.ones([h*rows, w*cols, ch])
    for i in range(rows):
        for j in range(cols):
            if i*cols+j < n:
                mosaic[i*h:(i+1)*h, (j*w):(j+1)*w, :] = samples[i*cols+j, ...]
                # plt.text((j*w), i*h+h//2, str(i*cols+j), color='r', fontsize=16)
    return mosaic.squeeze()


def to_image_8u(samples, w, h, ch=3):
    return (to_image(samples, w, h, ch)*255.99).astype(np.uint8)


def to_images_8u(samples, w, h, ch=3):
    return (to_images(samples, w, h, ch)*255.99).astype(np.uint8)


def to_cv_image(sample, w, h, ch=3):
    img = to_image_8u(sample, w, h, ch)
    if ch==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def to_cv_images(sample, w, h, ch=3):
    img = to_images_8u(sample, w, h, ch)
    if ch == 3:
        for i in range(img.shape[0]):
            img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)
    return img


def kmeans_clustering(samples, num_clusters, get_centers=False):
    N, d = samples.shape
    K = num_clusters
    # Select random d_used coordinates out of d
    d_used = min(d, max(500, d//8))
    d_indices = np.random.choice(d, d_used, replace=False)
    print('Performing k-means clustering to {} components of {} samples in dimension {}/{} ...'.format(K, N, d_used, d))
    with Timer('K-means'):
        clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(samples[:, d_indices])
    labels = clusters.labels_
    if get_centers:
        centers = np.zeros([K, d])
        for i in range(K):
            centers[i, :] = np.mean(samples[labels == i, :], axis=0)
        return labels, centers
    return labels


def gmm_initial_guess(samples, num_components, latent_dim, clustering_method='km', component_model='fa',
                      default_noise_std=0.5, dataset_std=1.0):
    N, d = samples.shape
    components = {}
    if clustering_method == 'rnd':
        # In random mode, l+1 samples are randomly selected per component, a plane is fitted through them and the
        # noise variance is set to the default value.
        print('Performing random-selection and FA/PPCA initialization...')
        for i in range(num_components):
            fa = FactorAnalysis(latent_dim)
            used_samples = np.random.choice(N, latent_dim + 1, replace=False)
            fa.fit(samples[used_samples])
            components[i] = {'A': fa.components_.T, 'mu': fa.mean_, 'D': np.ones([d])*np.power(default_noise_std, 2.0),
                             'pi': 1.0/num_components}
    elif clustering_method == 'km':
        # In k-means mode, the samples are clustered using k-means and a PPCA or FA model is then fitted for each cluster.
        labels = kmeans_clustering(samples/dataset_std, num_components)
        print("Estimating Factor Analyzer parameters for each cluster")
        components = {}
        for i in range(num_components):
            print('.', end='', flush=True)
            if component_model == 'fa':
                model = FactorAnalysis(latent_dim)
                model.fit(samples[labels == i])
                components[i] = {'A': model.components_.T, 'mu': model.mean_, 'D': model.noise_variance_,
                                 'pi': np.count_nonzero(labels == i)/float(N)}
            elif component_model == 'ppca':
                model = PCA(latent_dim)
                model.fit(samples[labels == i])
                components[i] = {'A': model.components_.T, 'mu': model.mean_, 'D': np.ones([d])*model.noise_variance_/d,
                                 'pi': np.count_nonzero(labels == i)/float(N)}
            else:
                print('Unknown component model -', component_model)
        print()
    else:
        print('Unknown clustering method -', clustering_method)
    return mfa.MFA(components)


def im_show(img):
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, vmin=0, vmax=1, cmap='gray')


def plot_samples(samples, image_size=(64, 64), rows=10, cols=10):
    ch = 3 if len(image_size)==2 else image_size[2]
    for i in range(min(rows*cols, samples.shape[0])):
        plt.subplot(rows, cols, i+1)
        im_show(to_image(samples[i, :], image_size[0], image_size[1], ch))
        plt.axis('off')
    plt.pause(0.1)


def visualize_component(gmm, image_shape=[32, 32], fig_num=1, c_begin=0, mean=0.0, std=1.0):
    h, w = image_shape[0:2]
    ch = 3 if len(image_shape)==2 else image_shape[2]
    K = min(8, len(gmm.components)-c_begin)
    l = min(10, gmm.components[0]['A'].shape[1])
    s = min(2.5, max(0.75, image_shape[0]/64.0))

    plt.figure(num=fig_num, figsize=(K*s*3, (l+1)*s))
    for i in range(K):
        c = gmm.components[c_begin+i]
        # MU and D on top of each column
        samples = np.ones([l+1, h, w*3+2, ch], dtype=float)
        samples[0, :, w//2:w//2+w, :] = (c['mu']*std + mean).reshape([h, w, ch])
        noise_std = np.sqrt(c['D'])
        noise_std /= np.max(noise_std)
        samples[0, :, w//2+w:w//2+2*w, :] = noise_std.reshape([h, w, ch])

        # Then the directions and the noise variance
        for j in range(l):
            samples[j+1, :, :w, :] = ((c['mu'] + 2 * c['A'][:, j])*std + mean).reshape([h, w, ch])
            samples[j+1, :, 2*(w+1):, :] = ((c['mu'] - 2 * c['A'][:, j])*std + mean).reshape([h, w, ch])
            samples[j+1, :, w+1:2*w+1, :] = ((0.5 + 2 * c['A'][:, j])*std + mean).reshape([h, w, ch])

        for j in range(samples.shape[0]):
            plt.subplot(samples.shape[0], K, j*K+i+1)
            image = np.maximum(0.0, np.minimum(1.0, samples))
            im_show(np.squeeze(image[j, ...]))
            plt.axis('off')


def visualize_component_samples(gmm, image_shape=[32, 32], fig_num=1, mean=0.0, std=1.0):
    h, w = image_shape[0:2]
    ch = 3 if len(image_shape)==2 else image_shape[2]
    K = min(16, len(gmm.components))
    s = min(2.5, max(0.75, image_shape[0]/64.0))
    m = 8
    # Plot samples from each component
    plt.figure(num=fig_num, figsize=(K*s, m*s))
    for i in range(K):
        c = gmm.components[i]
        samples = mfa.MFA()._draw_from_component(m, c, add_noise=False)
        samples = samples*std + mean
        samples = samples.reshape([samples.shape[0], h, w, ch])
        for j in range(samples.shape[0]):
            plt.subplot(samples.shape[0], K, j*K+i+1)
            image = np.maximum(0.0, np.minimum(1.0, samples))
            im_show(np.squeeze(image[j, ...]))
            plt.axis('off')


def visualize_random_samples(gmm, image_shape=[32, 32], fig_num=1, mean=0.0, std=1.0):
    h, w = image_shape[0:2]
    ch = 3 if len(image_shape)==2 else image_shape[2]
    s = min(2.5, max(0.75, image_shape[0]/64.0))
    m = 8
    # Draw random samples from the mixture
    plt.figure(num=fig_num, figsize=(m*s, m*s))
    samples = gmm.draw_samples(m*m, add_noise=False)
    samples = samples*std + mean
    samples = samples.reshape([samples.shape[0], h, w, ch])
    for j in range(samples.shape[0]):
        plt.subplot(m, m, j+1)
        image = np.maximum(0.0, np.minimum(1.0, samples[j, ...]))
        im_show(np.squeeze(image))
        plt.axis('off')


def visualize_trained_model(gmm, iter, image_shape=[32, 32], out_folder=None, mean=0.0, std=1.0, all_comps=False):
    if all_comps and out_folder:
        for c_i in range(0, len(gmm.components), 8):
            visualize_component(gmm, image_shape, fig_num=1, c_begin=c_i, mean=mean, std=std)
            plt.savefig(os.path.join(out_folder, 'comp_%d-%d_directions_%06d.jpg' % (c_i, c_i+7, iter)))
    else:
        visualize_component(gmm, image_shape, fig_num=1, mean=mean, std=std)
        if out_folder:
            plt.savefig(os.path.join(out_folder, 'comp_directions_%06d.jpg' % iter))

    visualize_component_samples(gmm, image_shape, fig_num=2, mean=mean, std=std)
    if out_folder:
        plt.savefig(os.path.join(out_folder, 'comp_samples_%06d.jpg' % iter))

    visualize_random_samples(gmm, image_shape, fig_num=3, mean=mean, std=std)
    if out_folder:
        plt.savefig(os.path.join(out_folder, 'rand_samples_%06d.jpg' % iter))

    plt.pause(0.1)


def kl_divergence(p, q):
    """
    The Kullback–Leibler divergence.
    Defined only if q != 0 whenever p != 0.
    """
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(q))
    assert not np.any(np.logical_and(p != 0, q == 0))

    p_pos = (p > 0)
    return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))


def jensen_shannon_divergence(p, q):
    """
    Calculates the symmetric Jensen–Shannon divergence between the two PDFs
    """
    m = (p + q) * 0.5
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


class Timer(object):
    def __init__(self, name='Operation'):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('%s took: %s sec' % (self.name, time.time() - self.tstart))


def get_dataset_mean_and_std(image_provider, num_samples=20000):
    mean_file_name = os.path.join(image_provider.image_folder, '../training_mean.npy')
    std_file_name = os.path.join(image_provider.image_folder, '../training_sgd.npy')
    if not os.path.isfile(mean_file_name):
        print('Calculating dataset mean and std...')
        samples = image_provider.get_random_samples(num_samples)
        dataset_mean = np.mean(samples, axis=0)
        dataset_std = np.std(samples - dataset_mean, axis=0)
        np.save(mean_file_name, dataset_mean)
        np.save(std_file_name, dataset_std)
    else:
        dataset_mean = np.load(mean_file_name)
        dataset_std = np.load(std_file_name)
    return dataset_mean, dataset_std
