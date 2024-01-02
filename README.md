# ICDAR 2024 Competition on Historical Map Text Detection, Recognition, and Linking

This repository contains the official evaluation implementation for the [ICDAR'24 MapText competition](https://rrc.cvc.uab.es/?ch=28).

Although closely related to previous competitions on robust reading (e.g., ICDAR19-ArT) and document layout analysis (e.g., ICDAR23-HierText), detecting and recognizing text on maps poses new challenges such as complex backgrounds, dense text labels, and various font sizes and styles. 
The competition features two primary tasks---text detection and end-to-end text recognition---each with a secondary task of linking words into phrase blocks.

## Installation

Installation depends on whether one is using Conda,  VirtualEnv with Pip, or Pipenv.
Note that the Robust Reading Challenge (RRC) server hosting the competition currently uses Python 3.9.18; 
although newer versions should also work, using the same version (as specified below) is more likely to produce matching results.

### Conda Installation
If using conda, create a conda environment with the requisite versions:

```shell
conda create -n maptext-eval python=3.9.18 --file conda-requirements.txt
conda activate maptext-eval
pip install -r pip-requirements.txt
```

### Pipenv Installation
Make sure you have [Pipenv](https://pipenv.pypa.io/) (along with [Pyenv](https://github.com/pyenv/pyenv)) installed.
Pipenv is a more mature virtual environment manager, and Pyenv enables to install any version of Python.

You can install all dependencies and open a shell in the new virtual environment with the following commands:
```shell
pipenv install
pipenv shell
```

### Virtualenv/Pip Installation
If using virtualenv, create a virtualenv environment and install the requisite versions:

```shell
virtualenv -p /usr/bin/python3.9 maptext-eval 
source maptext-eval/bin/activate
pip install -r requirements.txt
```

### Optional Apache Spark Installation
By default, the module can use Python's multiprocessing Pool to speed the evaluation calculation.
However, Apache Spark is also supported for those who may prefer it.
The Spark library must be installed.

Via conda:
```shell
conda install --file optional-requirements.txt
```

Or via pip:
```shell
pip install -r optional-requirements.txt
```

Or via pipenv:
```shell
pipenv install -r optional-requirements.txt
```

## Run

To run from the command-line:

```shell
python3 eval.py --gt GT.json --pred YOURFILE.json --task TASK
```
The options for `TASK` can be `det` (Task 1), `detlink` (Task 2), `detrec` (Task 3), or `detreclink` (Task 4).

To parallelize calculation, use `--parallel pool` for the built-in Python method or `--parallel spark` if using Apache Spark.

For other options (including writing per-image results to persistent file output) use `--help`.

| Output Key      | Metric                                 |
| :-------------  | :------------------------------------- |
| `det_quality`   | Panoptic Detection Quality (PDQ)       |
| `rec_quality`   | Panoptic Recognition Quality (PRQ)     |
| `char_quality`  | Panoptic Character Quality (PCQ)       |
| `char_accuracy` | 1 – Avg NED (Normalized Edit Distance) |

See [Competition Tasks](https://rrc.cvc.uab.es/?ch=28&com=tasks) for additional details and metric definitions.

### Examples

#### Included Example Data
Command:
```shell
python3 eval.py --gt data/example_gt.json --pred data/example_pred.json --task detrec
```
Output:
```json
{"det_recall": 0.875, "det_precision": 1.0, "det_fscore": 0.9333333333333333, "det_tightness": 0.7915550522475774, "det_quality": 0.7387847154310723, "rec_recall": 0.375, "rec_precision": 0.42857142857142855, "rec_fscore": 0.39999999999999997, "rec_tightness": 0.7638758923791641, "rec_quality": 0.3055503569516656, "char_accuracy": 0.8067226890756303, "char_quality": 0.5959943922805289}
```

#### Sample Data
After downloading sample data ([below](#data)) and producing predictions:

```shell
python3 eval.py --gt sample.json --pred YOUROUTPUT.json --task det --parallel pool --gt-regex sample
```

## Data

See competition [downloads](https://rrc.cvc.uab.es/?ch=28&com=downloads) for data details and [tasks](https://rrc.cvc.uab.es/?ch=28&com=tasks) for file format details.

* Small sample data set (353 tiles from 31 maps of 9 atlases) (DOI:[10.5281/zenodo.10444912](https://doi.org/10.5281/zenodo.10444912))
  - [Ground Truth JSON](https://zenodo.org/records/10444913/files/sample.json?download=1) (2.2M)
  - [Images ZIP](https://zenodo.org/records/10444913/files/sample.zip?download=1) (5.5G)
* Competition Validation Data (Expected 1 Feb 2024)
  - General Rumsey Data Set (100 tiles from 100 maps)
    * Ground Truth
    * Images
* Competition Training Data (Expected 1 Feb 2024)
  - General Rumsey Data Set (200 tiles from 200 maps)
    * Ground Truth
    * Images
  - French Cadastre Data Set (50 tiles from 50 maps)
    * Ground Truth
    * Images
* Competition Test Data (Expected 1 Mar 2024)
  - General Rumsey Data Set (700 tiles from 700 maps)
    * Images
  - French Cadastre Data Set (50 tiles from 50 maps)
    * Images

