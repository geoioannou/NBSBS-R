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






