# Installation of Cython.
echo "Performing the Cython Installation"
cd Configuration/ ; sh install.sh ; cd ..

mkdir Results;
mkdir Results/tuning-knn-3

# # Running each recommender in sequence, it may take more time but won't make
# # going out of space while using MovieLens10M, MovieLens20M or Netflix100M.
# cd run-examples/ ; sh holdout-cotraining-knn-funksvd.sh -p $PPOSITIVES -n $NNEGATIVES; cd ..
# cd run-examples/ ; sh holdout-cotraining-knn-bprmf.sh -p $PPOSITIVES -n $NNEGATIVES; cd ..
cd run-tuning/ ; sh knn.sh; cd ..
