# GOAL
This is a quick implementation of a CNN on the CIFAR-10 dataset using python JAX. The reasoning behind using JAX is that it allows for more precise control over the random seeds when training a model.

## Random Seeds
Under the "Initializating Model (RANDOM SEEDS)" section, you can see the different seeds in the progrma. 
- 'rng' is used for general purposes like splitting the dataset
- 'inp_rng' is what determines the min-batches selected during SGD
- 'init_rng' controls the initialization of the SGD, ie: it intializes the weights to some arbitrary numbers and adjusts after using SGD

## Utils.py
This file has two methods: one normalizes the image values, and the other converts them into numpy arrays
This was made into a separate file so all the workers in the data_loaders could access the methods.
