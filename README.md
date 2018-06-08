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
- `ndb_mnist_demo.py`: A small demo for NDB ealuation using MNIST (compare train, val, test and simulated biased model)

## Running the NDB MNIST Demo
- Download and extract MNIST from http://yann.lecun.com/exdb/mnist/ to `./data/mnist` (the folder should contain 4 `ubyte` files
- Run `python ndb_mnist_demo.py`:


```
Loading MNIST data...
Splitting MNIST training data to random train/val (45000/15000).
Initialize NDB bins with training samples
Performing K-Means clustering of 45000 samples in dimension 784 / 784 to 100 clusters ...
Can take a couple of minutes...
Done.
Evaluating 10000 validation samples (randomly split from the train samples - should be very similar)
Results for 10000 samples from Validation: NDB = 0 , JS = 0.00136090685292
Simulate a deviation from the data distribution (mode collapse) by sampling from specific labels (digits)
Results for 10000 samples from Val0-8: NDB = 11 , JS = 0.0159993090336
Evaluating 10000 test-set samples (different writers - can be somewhat different)
Results for 10000 samples from Test: NDB = 4 , JS = 0.00407072041308
```

The resulting binning histogram:
<img src="images/mnist_histograms_100.png"/>

A plot showing the NDB bin centers and the missing bins in the simulated Val0-8 evaluated set:
<img src="images/bins_with_Val0-8_results_100.png"/>
