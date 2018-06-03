import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
from matplotlib import pyplot as plt

class NDB:
    def __init__(self, training_data=None, number_of_bins=100, whitening=False, max_dims=None):
        self.number_of_bins = number_of_bins
        self.whitening = whitening
        self.ndb_eps = 1e-6
        self.training_mean = 0.0
        self.training_std = 1.0
        self.max_dims = max_dims
        self.bin_centers = None
        self.bin_proportions = None
        self.ref_sample_size = None
        self.cached_results = {}
        if training_data is not None:
            self.construct_bins(training_data)

    def construct_bins(self, training_samples):
        """
        Performs K-means clustering of the training samples
        :param training_samples: An array of m x d floats (m samples of dimension d)
        """
        n, d = training_samples.shape
        if self.whitening:
            self.training_mean = np.mean(training_samples, axis=0)
            self.training_std = np.std(training_samples, axis=0) + self.eps

        if self.max_dims is None and d > 1000:
            # To ran faster, perform binning on sampled data dimension (i.e. don't use all channels of all pixels)
            self.max_dims = d//6

        whitened_samples = (training_samples-self.training_mean)/self.training_std
        d_used = d if self.max_dims is None else min(d, self.max_dims)
        d_indices = np.random.permutation(d)

        print('Performing K-Means clustering of {} samples in dimension {} / {} ...'.format(n, d_used, d))
        clusters = KMeans(n_clusters=self.number_of_bins, max_iter=100, n_jobs=-1).fit(whitened_samples[:, d_indices[:d_used]])

        bin_centers = np.zeros([self.number_of_bins, d])
        for i in range(self.number_of_bins):
            bin_centers[i, :] = np.mean(whitened_samples[clusters.labels_ == i, :], axis=0)

        # Organize bins by size
        label_vals, label_counts = np.unique(clusters.labels_, return_counts=True)
        bin_order = np.argsort(-label_counts)
        self.bin_proportions = label_counts[bin_order] / np.sum(label_counts)
        self.bin_centers = bin_centers[bin_order, :]
        self.ref_sample_size = n
        print('Done. Reference bin proportions = ', self.bin_proportions)

    def calculate_score(self, query_samples, model_label=None):
        """
        Assign each sample to the nearest bin center (in L2). Pre-whiten if required. and calculate the NDB
        (Number of statistically Different Bins) and JS divergence scores.
        :param query_samples: An array of m x d floats (m samples of dimension d)
        :param model_label: optional label string for the evaluated model, allows plotting results of multiple models
        :return: results dictionary containing NDB and JS scores and array of labels (assigned bin for each query sample)
        """
        n = query_samples.shape[0]
        query_bin_proportions = self.__calculate_bin_proportions(query_samples)
        # print(query_bin_proportions)
        ndb = NDB.two_proportions_z_test(self.bin_proportions, self.ref_sample_size,
                                         query_bin_proportions, n)
        js = NDB.jensen_shannon_divergence(self.bin_proportions, query_bin_proportions)
        if model_label:
            print('Results for {} samples from {}: '.format(n, model_label), end='')
            self.cached_results[model_label] = {'NDB': ndb, 'JS': js, 'Proportions': query_bin_proportions, 'N': n}
        print('NDB =', ndb, ', JS =', js)

    def plot_results(self, models_to_plot=None):
        """
        Plot the binning proportions of different methods
        :param models_to_plot: optional list of model labels to plot
        """
        K = self.number_of_bins
        w = 1.0 / (len(self.cached_results)+1)
        assert K == self.bin_proportions.size
        assert self.cached_results

        # Used for plotting only
        def calc_se(p1, n1, p2, n2):
            p = (p1 * n1 + p2 * n2) / (n1 + n2)
            return np.sqrt(p * (1 - p) * (1/n1 + 1/n2))

        if not models_to_plot:
            models_to_plot = sorted(list(self.cached_results.keys()))

        train_se = calc_se(self.bin_proportions, self.ref_sample_size,
                                        self.bin_proportions, self.cached_results[models_to_plot[0]]['N'])
        plt.bar(np.arange(0, K)+0.5, height=train_se*2.0, bottom=self.bin_proportions-train_se,
                width=1.0, label='Train', color='gray')

        for i, model in enumerate(models_to_plot):
            results = self.cached_results[model]
            label = '%s (%i : %.4f)' % (model, results['NDB'], results['JS'])
            print(model, i, (i+0.25)*w, w)
            plt.bar(np.arange(0, K)+(i+1.0)*w, results['Proportions'], width=w, label=label)
        plt.legend(loc='best')
        plt.grid(True)
        plt.title('Binning Proportions Evaluation Results (NDB : JS)')
        plt.show()

    def __calculate_bin_proportions(self, samples):
        if self.bin_centers is None:
            print('First run construct_bins on samples from the reference training data')
        assert samples.shape[1] == self.bin_centers.shape[1]
        n = samples.shape[0]
        k = self.bin_centers.shape[0]
        D = np.zeros([n, k], dtype=samples.dtype)
        whitened_samples = (samples-self.training_mean)/self.training_std
        for i in range(k):
            D[:, i] = np.linalg.norm(whitened_samples - self.bin_centers[i, :], ord=2, axis=1)
        labels = np.argmin(D, axis=1)
        probs = np.zeros([k])
        label_vals, label_counts = np.unique(labels, return_counts=True)
        probs[label_vals] = label_counts / n
        return probs


    @staticmethod
    def two_proportions_z_test(p1, n1, p2, n2, significance_level=0.05):
        # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
        # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))    # Two-tailed test
        return np.count_nonzero(p_values < significance_level)

    @staticmethod
    def jensen_shannon_divergence(p, q):
        """
        Calculates the symmetric Jensen–Shannon divergence between the two PDFs
        """
        m = (p + q) * 0.5
        return 0.5 * (NDB.kl_divergence(p, m) + NDB.kl_divergence(q, m))

    @staticmethod
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


if __name__ == "__main__":
    d=100
    k=50
    n_train = k*100
    n_test = k*10

    training_samples = np.random.uniform(size=[n_train, d])
    ndb = NDB(training_data=training_samples, number_of_bins=k)

    test_samples = np.random.uniform(high=1.0, size=[n_test, d])
    ndb.calculate_score(test_samples, model_label='Test')

    test_samples = np.random.uniform(high=0.9, size=[n_test, d])
    ndb.calculate_score(test_samples, model_label='Good')

    test_samples = np.random.uniform(high=0.75, size=[n_test, d])
    ndb.calculate_score(test_samples, model_label='Bad')

    ndb.plot_results(models_to_plot=['Test', 'Good', 'Bad'])
