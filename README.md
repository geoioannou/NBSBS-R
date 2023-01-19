# Online Batch Selection for Enhanced Generalization in Imbalanced Datasets

This is the repository for the implementation of NBSBS-R.


## Training

First, you have to prepare the folders with the datasets and their transformed versions. An example of the structure of the folders should be like that (for adasyn):

```
├───adasyn
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone

```
Inside each subfolder insert the trainsets and testsets to continue.

To start the training just set the parameters in script.py and execute:

```
python script.py
```
Hyperparameters to tune:
* swapped: [0, 0.2]
* re-enter: {0, 1}
* noisy: {0, 1}


## Requirements
* python == 3.7.9
* tensorflow == 2.4.0
* numpy == 1.19.5
* scikit-learn == 1.0.2





Citation:

If you plan to use NBSBS-R in your reseach, please cite the following journal article:
```
@Article{a16020065,
AUTHOR = {Ioannou, George and Alexandridis, Georgios and Stafylopatis, Andreas},
TITLE = {Online Batch Selection for Enhanced Generalization in Imbalanced Datasets},
JOURNAL = {Algorithms},
VOLUME = {16},
YEAR = {2023},
NUMBER = {2},
ARTICLE-NUMBER = {65},
URL = {https://www.mdpi.com/1999-4893/16/2/65},
ISSN = {1999-4893},
ABSTRACT = {Importance sampling, a variant of online sampling, is often used in neural network training to improve the learning process, and, in particular, the convergence speed of the model. We study, here, the performance of a set of batch selection algorithms, namely, online sampling algorithms that process small parts of the dataset at each iteration. Convergence is accelerated through the creation of a bias towards the learning of hard samples. We first consider the baseline algorithm and investigate its performance in terms of convergence speed and generalization efficiency. The latter, however, is limited in case of poor balancing of data sets. To alleviate this shortcoming, we propose two variations of the algorithm that achieve better generalization and also manage to not undermine the convergence speed boost offered by the original algorithm. Various data transformation techniques were tested in conjunction with the proposed scheme to develop an overall training method of the model and to ensure robustness in different training environments. An experimental framework was constructed using three naturally imbalanced datasets and one artificially imbalanced one. The results assess the advantage in convergence of the extended algorithm over the vanilla one, but, mostly, show better generalization performance in imbalanced data environments.},
DOI = {10.3390/a16020065}
}
```

