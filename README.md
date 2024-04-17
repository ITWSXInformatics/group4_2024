### Frequency domain information Xgboost classification on TEP dataset

Considering Xgoost's incapability to deal with multi-timestamp time series data, we proposed frequency domain information extraction method to distill the feature from a multi-timestamp time series, taking it as input of Xgboost. Specically, we use wavelet decomposition method to split the information from time series into many different frequency message, and calculate mean and std for every different frequency chop, taking them as input of classifier, i.e. Xgboost.

### task description

We use TEP dataset to build an abnormality classification model. Specifically, we train model only on abnormal data (all dat except d00.dat and d00_te.dat, which are normal data).

We compare our proposed model with time domain Xgboost classification to prove that our model can improve classfication performance significantly. The result shows the proposed model imporve accuracy by nearly 15%

### data description

1. For both dataset(tep_train and tep_test):

- d00.dat is normal dataset, and the rest are abnormal with different type of abormality

2. training data (tep_train):

- length is 480, and all are abnormal

3. testing data (tep_test)

- length is 960, but the former 160 are normal and latter 800 are abnormal, so we only use latter 800, which is abnormal
- divide 800 into 300/500 as validation/testing dataset

### How to run:

Using python:
python main.py tep_train tep_test [time_step, default=110] [level, default=6] [--wavelet, default=True]

Using shell:
shell run.sh

### Checkpoint:

The pretrained model could be found in: https://drive.google.com/file/d/13JZFrSf6L-hA_mXdAxyguBhEaX-z5pYD/view?usp=drive_link

### Interface:
Process to Start the Repository:

Create a new directory for your project and navigate to it in the terminal. Create a virtual environment (optional but recommended):

python -m venv venv

source venv/bin/activate Install the required dependencies:

pip install flask numpy pandas xgboost
