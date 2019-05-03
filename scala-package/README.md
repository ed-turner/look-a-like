# Welcome to the Look-A-Like Scala Package!

Author: Edward Turner

## Introduction

Generally, we want to be able to predict various characteristics,
 perhaps simultaneously, ensuring that the samples in the testing
  dateset that "looks like" the samples in the training dataset have 
 similar predictive values.  There are various methods that exist 
 today that are predictive in nature, and are well documented. However, 
 there are few that is able to ensure that samples from the testing dataset 
 with similar features as in the testing dataset have similar predictive values.
 
 This python package delivers a highly sought-after methodology, which utilizes
 the relative importance each feature has to be predictive to our chosen value
 and scales our features accordingly their importance, then perform a nearest 
 neighbors algorithm to generate our matches.  
 
 A more full description of the methodology is found under the Methodology section.

 This package is apache-spark compatible!  So do not worry about developing matches
 using big data.  This package has that covered!

## Installation

TODO

## Methodology

As mentioned in the introduction, we derive some values that are based on
the predictive power of each feature and scale those features by those values. To
do that, we use the Light Gradient Boosting Method (LGBM) to fit the training dataset. 
To optimize the LGBM using bayesian hyper parameter optimize on a train/validation
split on the original training dataset.  Once optimized, we fit on the entire 
training dataset. By doing so, we will generate the feature importance for 
each feature.  We then scale our feature importance so that they are nonzero 
and sum to one.  This is the very first step.  

Once we derive our feature importance, we scale our features according to their
feature importance, after standardizing our features.  There are several available
distance measures to use for our matching algorithm, along with different ways
to find our closest neighbors.  For our distance calculation, we have the 
p-norm measure and the cosine measure. For our nearest-neighbors algorithm, we 
have the k-nearest-neighbors algorithm and the hungarian-matching algorithm. 
This gives us a total of 4 types of matching algorithms.  

## Tutorial

TODO

## Documentation

For code documentations, please go to https://look-a-like.github.io/scala

Or have a look at the code repository.

