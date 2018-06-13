On GANs and GMMs
================
Implementation of *NDB* and *MFA* per [On GANs and GMMs](https://arxiv.org/abs/1805.12462) by Eitan Richardson and Yair Weiss.

**NDB**: An evaluation method for high-dimensional generative models

**MFA**: Mixture of Factor Analyzer for modeling high-dimensional data (e.g. full images)

### TODO:
- [x] NDB code cleanup
- [x] MNIST demo for NDB
- [ ] MFA model cleanup

### Prerequisites

- Python 3.x, NumPy, SciPy, Sklearn

### Models

- `ndb.py`: The NDB evaluation method class
- `ndb_mnist_demo.py`: A small demo for NDB evaluation using MNIST (compare train, val, test and simulated biased model)

## Running the NDB MNIST Demo
- Download and extract MNIST from http://yann.lecun.com/exdb/mnist/ to `./data/mnist` (should contain 4 `ubyte` files)
- Run `python ndb_mnist_demo.py`:

The resulting binning histogram and NDB (Number of statistically Different Bins) and JS (Jensen-Shannon Divergence) values:
<img src="images/mnist_histograms_100.png"/>

NDB evaluation on this toy example reveals that:
- A random validation split from the train data is statistically similar to the train data (NDB = 0, JS divergence = 0.0014)
- The MNIST test set is not coming from exactly the same distribution (different writers), but is pretty close (NDB = 4, JS divergence = 0.0041)
- NDB detects the distribution distortion in a deliberately biased set created by removing all digit 9 samples from the validation sets (NDB = 11, JS divergence = 0.016)

A plot showing the NDB bin centers and the missing bins in the simulated Val0-8 evaluated set (all bins corresponding to digit 9):
<img src="images/bins_with_Val0-8_results_100.png"/>


