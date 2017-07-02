# Installation of Conda and the cotraining environment if it doesn't exist.

# Installation of Cython.
cd Configuration/ ; sh install.sh ; cd ..

# Running each recommender in sequence, it may take more time but won't make
# going out of space while using MovieLens10M, MovieLens20M or Netflix100M.
cd run-examples/ ; sh holdout-cotraining-knn-knn.sh ; cd ..
cd run-examples/ ; sh holdout-cotraining-knn-funksvd.sh ;cd ..
cd run-examples/ ; sh holdout-cotraining-mf-mf.sh ; cd ..
