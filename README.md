# Improving Collaborative Filtering Techniques by the use of Co-Training in Recommender Systems. #
This repository holds the implementation, the datasets and results for the thesis project: Improving Collaborative Filtering Techniques by the use of Co-Training in Recommender Systems.

This project was done under a thesis research for Fernando Benjamín Pérez Maurera, under supervision of Professor Paolo Cremonesi and
Engineer Maurizio Ferrari, at Politecnico di Milano.

## Project organization. ##
The project is organized as follows:
  - Datasets: A folder where the datasets are located.
    - ml10m: folder containing the Movielens10M dataset.
        - ratings.csv: ratings file.
  - Implementation: RecPy module where the recommenders and helper classes are.
  - read-results: folder where `bash` scripts are, and are used to read the results output by Co-Training. There are for each recommender combination. The cases inside are `ItemKNN/FunkSVD`, `ItemKNN/SLIM`, `ItemKNN/SLIMBPR` and `ItemKNN/BPRMF`.
  - run-examples: folder where `bash` scripts to run Co-Training are. There are several recommenders combinations as: `ItemKNN/FunkSVD`, `ItemKNN/SLIM`, `ItemKNN/SLIMBPR` and `ItemKNN/BPRMF`.
  - Results: A folder generated when running `run-knn.sh` and `results-knn/sh` scripts. In this folder the results for each test-case will be put.
  - scripts: Folder where its located the two main `Python` files, `holdout.py`, which makes a holdout@k of the dataset, runs Co-Training and evaluates the recommenders, and `read-results.py`, which reads the results of each output file and generates new plots.
  - `README.md`: This file.
  - `requirements.txt`: File for Conda or PIP that has the libraries and modules required to run the code.
  - `results-knn.sh`: main `bash` script to read the results that each Co-Training process outputs for each test case inside `read-results`.
  - `run-knn.sh`: main `bash` script to run the Co-Training process for each test case inside `run-examples`.

## Project installation. ##
### Requirements.
  - `Python 3.6+`.
  - `C++` Compiler.
  - On Linux, ensure that you have packages `libc6-dev` and `build-essentials`

### Installation instructions
  1. [On Linux] Install Linux packages: `apt-get install -y libc6-dev build-essentials`.
  2. Install `Miniconda` for `Python 3.6+` [here](https://conda.io/miniconda.html).
  3. Create the virtual environment: `conda create -n cotraining --file requirements.txt`
  4. Activate the virtual environment: `source activate cotraining`.
  5. [Installation and run separately] Install the project: `cd Configuration/ ; sh install.sh ; cd ..`
  6. [Installation and run separately] Run one of the examples:
    * `cd run-examples/ ; sh knn-funksvd.sh -p <p-most positive> -n <n-most negative> -u <size of U'>; cd ..`
    * `cd run-examples/ ; sh knn-slim.sh -p <p-most positive> -n <n-most negative> -u <size of U'>; cd ..`
    * `cd run-examples/ ; sh knn-bprmf.sh -p <p-most positive> -n <n-most negative> -u <size of U'>; cd ..`
    * `cd run-examples/MyMediaLite/bin/ ; sh knn-slimbpr.sh  -p <p-most positive> -n <n-most negative> -u <size of U'>; cd ..`

  7. [Installation and run integrated] Run the `run-knn.sh` script: `sh run-knn.sh -p <p-most positive> -n <n-most negative> -u <size of U'>`
  8. [Only to generate new plots] Run the `results-knn.sh` script: `sh results-knn.sh -p <p-most positive> -n <n-most negative> -u <size of U'>`

## Results. ##
The test cases included with the project are `ItemKNN/FunkSVD`, `ItemKNN/SLIM`, `ItemKNN/SLIMBPR` and `ItemKNN/BPRMF`. The dataset used is Movielens10M, a holdout technique at 20% was used. A top-10 recommendation list was generated at evaluation time for each user. The items were divided into 10 bins based on their popularity, where the least popular is `bin_0` and the most popular is `bin_9`. When running the test cases, the `Results` folder will be created, and a subfolder for each test case will be created.

At the moment, the project generates output files for: evaluation of `RMSE`, `MAP`, `ROC-AUC`, `Precision`, `Recall`, `NDCG` and `MRR`, a file containing the number of p-most and n-most items rated at each iteration, the agreement between the recommenders, and the popularity of the items recommended.
