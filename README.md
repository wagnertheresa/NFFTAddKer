# NFFTAddKer
This repository contains the implementation of the framework described in the paper "Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods, and Kernel Derivatives".

## Usage
The main file [FE_nfft_kernel_ridge.py](https://github.com/wagnertheresa/NFFTAddKer/blob/main/FE_nfft_kernel_ridge.py) consists of the following two classes:
- `NFFTKernelRidgeFE` performs a NFFT-accelerated KRR on additive kernels.
- `GridSearch` searches on candidate parameter values for the classifiers `NFFTKernelRidge` or `sklearn KRR`.

[feature_engineering.py](https://github.com/wagnertheresa/NFFTAddKer/blob/main/feature_engineering.py) is the file in which all feature arrangement techniques are implemented.

## Data sets
The benchmark data sets used in for the numerical results can be downloaded from the following websites: [Protein](https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure), [KEGGundir](https://archive.ics.uci.edu/dataset/221/kegg+metabolic+reaction+network+undirected), [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and [Housing](http://lib.stat.cmu.edu/datasets/). The data files can also found in the [data](https://github.com/wagnertheresa/NFFTAddKer/tree/main/data) folder of this repository.

## References
This repository uses the [prescaledFastAdj](https://github.com/wagnertheresa/prescaledFastAdj) package to perform NFFT-accelerated kernel evaluations for the Gaussian and the Matérn(1/2) kernels and their derivative kernels.

## Citation
```
@article{wagner2024NFFTAddKer,
  title     = {Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods, and Kernel Derivatives},
  author    = {Theresa Wagner and Franziska Nestler and Martin Stoll},
  keywords  = {additive kernels, feature grouping, Fourier analysis, kernel derivatives, multiple kernel learning},
  url       = {https://arxiv.org/},
  year      = {2024}
}
```
