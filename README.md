# Online Batch Selection for Enhanced Generalization in Imbalanced Datasets

This is the repository for the implementation of NBSBS-R.


## Training

First, you have to prepare the folders with the datasets and their transformed versions. The structure of the folders should be like that:

```
├───adasyn
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───borderlinesmote
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───clust_centroids
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───enn
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───kmeanssmote
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───normal
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───ros
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───rus
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───smote
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───smoteenn
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
├───smotetomek
│   ├───adult
│   ├───credit
│   ├───mnist
│   │   └───1_minor_0.1
│   └───Ozone
└───tomeklinks
    ├───adult
    ├───credit
    ├───mnist
    │   └───1_minor_0.1
    └───Ozone
```



To start the training just set the parameters in script.py and execute:

```
python script.py
```

## Requirements
* python == 3.7.9
* tensorflow == 2.4.0
* numpy == 1.19.5
* scikit-learn == 1.0.2






