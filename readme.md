## Overview

This is an English translation of the sample code that accompaines the bestselling Japanese Kaggle book "Data Analysis Techniques to Win Kaggle" ([Amazon Japan](https://www.amazon.co.jp/dp/4297108437)). 
The book was written by Daisuke Kadowaki ([threecourse](https://www.kaggle.com/threecourse)), Ryuji Sakata ([Jack](https://www.kaggle.com/rsakata)), Keisuke Hosaka ([hskksk](https://www.kaggle.com/hskksk)) and Yuji Hiramatsu ([maxwell](https://www.kaggle.com/maxwell110)). 
It was first published on 9 October 2019 by Gijutsu-Hyohron Co., Ltd (ISBN-13: 978-4297108434).

Book cover:

<img src="misc/cover_small.jpg" width="200">

### Contents of each folder

| Folder | Contents |
|:----|:-------|
| input | Input files |
| ch01 | Sample code for chapter 1 |
| ch02 | Sample code for chapter 2 |
| ch03 | Sample code for chapter 3 |
| ch04 | Sample code for chapter 4 |
| ch05 | Sample code for chapter 5 |
| ch06 | Sample code for chapter 6 |
| ch07 | Sample code for chapter 7 |
| ch04-model-interface | Code for the classes and folder structure composition for competitions discussed in chapter 4 |

* Execute code with the each chapter folder directory as the current directory.
* For chapter 1, download the titanic data first as described in [input/readme.md](input/readme.md).
* For the chapter 4 model interface code, refer to [ch04-model-interface/readme.md](ch04-model-interface).


### Requirements

The sample code has been checked for operability on Google Cloud Platform (GCP) using the following environment.

* Ubuntu 18.04 LTS  
* Anaconda 2019.03 Python 3.7
* Necessary Python packages (check script below)

Use following script to set up GCP environment.
```
# utils -----

# Install required tools for development
cd ~/
sudo apt-get update
sudo apt-get install -y git build-essential libatlas-base-dev
sudo apt-get install -y python3-dev

# anaconda -----

# Download and install Anaconda
mkdir lib
wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh -O lib/anaconda.sh
/bin/bash lib/anaconda.sh -b

# Add to PATH
echo export PATH=~/anaconda3/bin:$PATH >> ~/.bashrc
source ~/.bashrc

# python packages -----

# Install Python packages
# Use Anaconda 2019.03 default versions for numpy, scipy and pandas
# pip install numpy==1.16.2 
# pip install scipy==1.2.1 
# pip install pandas==0.24.2
pip install scikit-learn==0.21.2

pip install xgboost==0.81
pip install lightgbm==2.2.2
pip install tensorflow==1.14.0
pip install keras==2.2.4
pip install hyperopt==0.1.1
pip install bhtsne==0.1.9
pip install rgf_python==3.4.0
pip install umap-learn==0.3.9

# set backend for matplotlib to Agg -----

# To execute on GCP, set matplotlib to backend
matplotlibrc_path=$(python -c "import site, os, fileinput; packages_dir = site.getsitepackages()[0]; print(os.path.join(packages_dir, 'matplotlib', 'mpl-data', 'matplotlibrc'))") && \
sed -i 's/^backend      : qt5agg/backend      : agg/' $matplotlibrc_path
```
