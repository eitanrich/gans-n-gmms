On GANs and GMMs
================
Implementation of *NDB* and *MFA* per [On GANs and GMMs](https://arxiv.org/abs/1805.12462) by Eitan Richardson and Yair Weiss.

**NDB**: An evaluation method for high-dimensional generative models

**MFA**: Mixture of Factor Analyzers for modeling high-dimensional data (e.g. full images)

### TODO:
- [x] NDB code cleanup
- [x] MNIST demo for NDB
- [x] MFA model cleanup
- [ ] Inference code

### Prerequisites

- Python 3.x, NumPy, SciPy, Sklearn, TensorFlow

### Demo Scripts

- `ndb_mnist_demo.py`: A small stand-alone demo for NDB evaluation using MNIST (compare train, val, test and simulated biased model)
- `mfa_train_celeba.py`: Training an MFA model for the CelebA dataset

## Running the NDB MNIST Demo
- Download and extract MNIST from http://yann.lecun.com/exdb/mnist/ to `./data/mnist` (should contain 4 `ubyte` files)
- Run `python3 ndb_mnist_demo.py`:

The resulting binning histogram, and NDB (Number of statistically Different Bins) and JS (Jensen-Shannon Divergence) values:

<img src="images/mnist_histograms_100.png"/>

NDB evaluation on this toy example reveals that:
- A random validation split from the train data is statistically similar to the train data (NDB = 0, JS divergence = 0.0014)
- The MNIST test set is not coming from exactly the same distribution (different writers), but is pretty close (NDB = 4, JS divergence = 0.0041)
- NDB detects the distribution distortion in a deliberately biased set created by removing all digit 9 samples from the validation set (NDB = 11, JS divergence = 0.016)

A plot showing the NDB bin centers and the statistically-different bins (with significance level = 3 standard-errors) in the simulated Val0-8 evaluated set. 
All bins corresponding to digit 9 are detected as statistically-different (marked in red).

<img src="images/bins_with_Val0-8_results_100.png"/>

**The NDB test can be used to evaluate different generative models (e.g. GANs). In our paper, we demonstrate its performance on three datasets - CelebA, SVHN and MNIST.**

## Training MFA on CelebA

First, download the CelebA dataset from <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html> into some folder (default location is ../../Datasets/CelebA/).
The CelebA folder should contain the subfolder img_align_celeba (containing all aligned and cropped images) and the text file list_eval_partition.txt.

Run `python3 mfa_train_celeba.py` ...

By Default, the script learns a Mixture of Factor Analyzers with 200 components and then breaks each component to additional components.
While running, the code generates some plots that are also saved to the output folder (sorry, no TensorBoard...):

The first few components, showing each component mean, noise std and column-vectors of the scale matrix:

<img src="images/comp_directions_009999.jpg"/>

Random samples drawn from each component separately:

<img src="images/comp_samples_009999.jpg"/>

Random samples from the mixture:

<img src="images/rand_samples_009999.jpg"/>

The test log-likelihood progress:

### Sub-components training

After training the main (root) components, the script continues to train sub-components (training is done hierarchically due to GPU memory limitations).

Below are examples for images generated for sub-components of one of the root components:

Components directions:

<img src="images/comp_directions_004999.jpg"/>

Random samples drawn from each sub-component separately:

<img src="images/comp_samples_004999.jpg"/>

Random samples from the mixture:

<img src="images/rand_samples_004999.jpg"/>

The test log-likelihood progress:


