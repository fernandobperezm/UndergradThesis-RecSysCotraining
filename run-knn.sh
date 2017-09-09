#!/bin/sh
# Poltecnico di Milano.
# run-knn.sh
# Description: This file runs the examples located inside the 'run-examples'
#               folder with their respective parameters.
# Created by: Fernando Benjamín Pérez Maurera.
# Last Modified: 09/09/2017.

# The options are: -p <number> -n <number> -u <number>
# -p represents the number of positive examples to label.
# -n represents the number of negative examples to label.
# -u represents the size of the pool of unlabeled samples.
while getopts p:n:u: option
do
    case "${option}"
        in
        p) PPOSITIVES=${OPTARG};;
        n) NNEGATIVES=${OPTARG};;
        u) UNLABELED=${OPTARG};;
    esac
done

# Installation of Cython.
echo "Performing the Cython Installation"
cd Configuration/ ; sh install.sh ; cd ..

mkdir Results;
mkdir Results/knn-funksvd-3; mkdir Results/knn-slimmt-3; mkdir Results/knn-slimbpr-3; mkdir Results/knn-bprmf-3;

# Running each recommender in sequence, it may take more time but won't make
# going out of space while using MovieLens10M, MovieLens20M or Netflix100M.
cd run-examples/ ; sh knn-funksvd.sh -p $PPOSITIVES -n $NNEGATIVES -u $UNLABELED; cd ..
cd run-examples/ ; sh knn-bprmf.sh -p $PPOSITIVES -n $NNEGATIVES -u $UNLABELED; cd ..
cd run-examples/ ; sh knn-slim.sh -p $PPOSITIVES -n $NNEGATIVES -u $UNLABELED; cd ..
cd run-examples/MyMediaLite/bin/ ; sh knn-slimbpr.sh -p $PPOSITIVES -n $NNEGATIVES -u $UNLABELED; cd ..
