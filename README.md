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
conda env create -f conda-environment.yaml
conda activate maptext-eval
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

| Output Key      | Metric                                                       |
| :-------------  | :----------------------------------------------------------- |
| `quality`       | Panoptic Detection or Recognition Quality (PDQ or PRQ)       |
| `char_quality`  | Panoptic Character Quality (PCQ)                             |
| `char_accuracy` | 1 – Avg NED (Normalized Edit Distance) among true positives  |

Note that `char_quality` and `char_accuracy` are only meaningful for the `detreclink` task.

See [Competition Tasks](https://rrc.cvc.uab.es/?ch=28&com=tasks) for additional details and metric definitions.

### Examples

#### Included Example Data
Command:
```shell
python3 eval.py --gt data/example_gt.json --pred data/example_pred.json --task detrec
```
Output:
```json
{"recall": 0.375,
 "precision": 0.42857142857142855,
 "fscore": 0.39999999999999997,
 "tightness": 0.7638758904606489,
 "quality": 0.3055503561842595,
 "char_accuracy": 1.0,
 "char_quality": 0.3055503561842595}
```

#### Sample Data
After downloading sample data ([below](#data)) and producing predictions:

```shell
python3 eval.py --gt sample.json --pred YOUROUTPUT.json \
  --task det --parallel pool --gt-regex sample
```

## Data

See competition [downloads](https://rrc.cvc.uab.es/?ch=28&com=downloads) for data details and [tasks](https://rrc.cvc.uab.es/?ch=28&com=tasks) for file format details.

* [General Rumsey Data Set ![DOI:10.5281/zenodo.10608900](https://zenodo.org/badge/DOI/10.5281/zenodo.10608900.svg)](https://doi.org/10.5281/zenodo.10608900)
  - Train Split (200 tiles from 196 maps)
  - Validation Split (40 tiles from 40 maps)
* [French Cadastre Data Set ![DOI:10.5281/zenodo.10610731](https://zenodo.org/badge/DOI/10.5281/zenodo.10610731.svg)](https://doi.org/10.5281/zenodo.10610731)
  - Train Split (80 tiles from 37 maps)
  - Validation Split (15 tiles from 9 maps)
* Competition Test Data
  - [General Rumsey Data Set ![DOI:10.5281/zenodo.10776182](https://zenodo.org/badge/DOI/10.5281/zenodo.10776182.svg)](https://doi.org/10.5281/zenodo.10776182) (700 tiles from 700 maps)
  - [French Cadastre Data Set ![DOI:10.5281/zenodo.10732280](https://zenodo.org/badge/DOI/10.5281/zenodo.10732280.svg)](https://doi.org/10.5281/zenodo.10732280) (50 tiles from 49 maps)
* [Sample Data Set ![DOI:10.5281/zenodo.10444912](https://zenodo.org/badge/DOI/10.5281/zenodo.10444912.svg)](https://doi.org/10.5281/zenodo.10444912)
  - 353 tiles from 31 maps of 9 atlases
  

