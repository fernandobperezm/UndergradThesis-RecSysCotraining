# # Installation of libc6-dev and build-essentials packages on Linux.
# # Will ask for permission.
# echo "Determining the OS type..."
# OS=$(uname)
# if [ $OS = "Linux" ]
# then
#     # Check if the packages are already installed.
#     { dpkg -l | grep 'libc6-dev'; } > /dev/null 2>&1
#     LIBC6_INSTALLED=$?
#     { dpkg -l | grep 'build-essential'; } > /dev/null 2>&1
#     BUILD_ESS_INSTALLED=$?
#     if [ $LIBC6_INSTALLED -ne 0 -o $BUILD_ESS_INSTALLED -ne 0 ]
#     then
#         (sudo apt-get install -y libc6-dev build-essentials)
#     fi
# elif [ $OS = "Darwin" ]
# then
#     echo "...OS is MacOS and those packages are not needed."
# fi
#
# # Installation of Conda and the cotraining environment if it doesn't exist.
# echo "Checking if miniconda is already installed..."
# MINICONDA_DIR="~/miniconda3"
# if [ -d "$MINICONDA_DIR" ];
# then
#     echo "Downloading conda installation script."
#     if [ $OS = "Linux" ]
#     then
#         # Determining the processor type, 32Bits or 64Bits.
#         PROC_TYPE=$(getconf LONG_BIT)
#         if [ $PROC_TYPE == 64 ]
#         then
#             wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#             bash Miniconda3-latest-Linux-x86_64.sh -b
#         else
#             wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh
#             bash Miniconda3-latest-Linux-x86.sh -b
#         fi
#     else
#         wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
#         bash Miniconda3-latest-MacOSX-x86_64.sh -b
#     fi
#
# else
#     echo "...Conda already exists."
# fi
#
# # Checking if the cotraining environment exists.
# echo "Checking if the Co-Training environment already exits..."
# CONDA_ENV="cotraining"
# { conda-env list | grep $CONDA_ENV; } > /dev/null 2>&1
# if [ $? -ne 0 ] # 0 means the conda environment exists. otherwise, another int.
# then
#     { conda create -n cotraining --file requirements.txt; } > /dev/null 2>&1
#     echo "...Successfully created the 'cotraining' environment."
# else
#     echo "...The environment 'cotraining' already existed."
# fi
#
# echo "Activating the environment."
# { source activate cotraining; } > /dev/null 2>&1

# Installation of Cython.
echo "Performing the Cython Installation"
cd Configuration/ ; sh install.sh ; cd ..

# mkdir Results;
# mkdir Results/knn-funksvd-1; mkdir Results/knn-funksvd-2; mkdir Results/knn-funksvd-3;
# mkdir Results/knn-knn-1; mkdir Results/knn-knn-2; mkdir Results/knn-knn-3; mkdir Results/knn-knn-4; mkdir Results/knn-knn-5; mkdir Results/knn-knn-6; mkdir Results/knn-knn-7;
# mkdir Results/knn-slim-1; mkdir Results/knn-slim-2; mkdir Results/knn-slim-3;
# mkdir Results/mf-mf-1; mkdir Results/mf-mf-2;
# mkdir Results/slim-funksvd-1;
# mkdir Results/slim-slim-1;

#
# # Running each recommender in sequence, it may take more time but won't make
# # going out of space while using MovieLens10M, MovieLens20M or Netflix100M.
# cd run-examples/ ; sh holdout-cotraining-knn-knn.sh ; cd ..
cd run-examples/ ; sh holdout-cotraining-knn-funksvd.sh ;cd ..
# cd run-examples/ ; sh holdout-cotraining-knn-slim.sh ; cd ..
# cd run-examples/ ; sh holdout-cotraining-slim-funksvd.sh ; cd ..
# cd run-examples/ ; sh holdout-cotraining-slim-slim.sh ; cd ..
# cd run-examples/ ; sh holdout-cotraining-mf-mf.sh ; cd ..
