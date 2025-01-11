# ICDAR 2025 Competition on Historical Map Text Detection, Recognition, and Linking

This repository contains the official evaluation implementation for the [ICDAR'25 MapText competition](https://rrc.cvc.uab.es/?ch=32).

(Evaluation for the [ICDAR'24 MapText
competition](https://rrc.cvc.uab.es/?ch=28) is available through repository tag `icdar-2024`.)

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

## Run

To run from the command-line:

```shell
python3 eval.py --gt GT.json --pred YOURFILE.json --task TASK
```
The options for `TASK` can be `det` (Task 1), `detedges` (Task 2), `detrec` (Task 3), or `detrecedges` (Task 4).

For other options (including writing per-image results to persistent file output) use `--help`.

| Output Key         | Metric                                                                     | Tasks                     |
| :----------------  | :------------------------------------------------------------------------- | :-------------------------|
| `recall`           | Fraction of ground truth words (not ignored) that are true positives       | all                       |
| `precision`        | Fraction of predicted words (not ignored) that are true positives          | all                       |
| `fscore`           | Harmonic mean of `recall` and `precision`                                  | all                       |
| `tightness`        | Average IoU among true positive words                                      | all                       |
| `quality`          | Panoptic Quality (PQ); product of `fscore` and `tightness`                 | all                       |
| `char_accuracy`    | 1 – Avg NED (Normalized Edit Distance) among true positive words           | `detrec`, `detrecedges`   |
| `char_quality`     | Panoptic Character Quality (PCQ); product of `quality` and `char_accuracy` | `detrec`, `detrecedges`   |
| `edges_recall`     | Fraction of ground truth word links (not ignored) that are true positives  | `detedges`, `detrecedges` |
| `edges_precision`  | Fraction of predicted word links (not ignored) that are true positives     | `detedges`, `detrecedges` |
| `edges_fscore`     | Harmonic mean of `edges_recall` and `edges_precision`                      | `detedges`, `detrecedges` |
| `hmean`            | Harmonic mean of all quantities for task-specific evaluation               | all                       |


See [Competition Tasks](https://rrc.cvc.uab.es/?ch=32&com=tasks) for additional details and metric definitions.

### Examples

#### Included Example Data
Command:
```shell
python3 eval.py --gt data/example_gt.json --pred data/example_pred.json --task detrecedges
```
Output:
```json
{"recall": 0.875, 
 "precision": 1.0, 
 "fscore": 0.9333333333333333, 
 "tightness": 0.7915550491751623, 
 "quality": 0.7387847125634848, 
 "hmean": 0.6939804663478768, 
 "char_accuracy": 0.8067226890756303, 
 "char_quality": 0.595994389967181, 
 "edges_recall": 0.3333333333333333, 
 "edges_precision": 1.0, 
 "edges_fscore": 0.5}
```

#### Sample Data
After downloading sample data ([below](#data)) and producing predictions:

```shell
python3 eval.py --gt sample.json --pred YOUROUTPUT.json \
  --task det --gt-regex sample
```

## Data

See competition [downloads](https://rrc.cvc.uab.es/?ch=32&com=downloads) for data details and [tasks](https://rrc.cvc.uab.es/?ch=32&com=tasks) for file format details.

* [General Rumsey Data Set ![DOI:10.5281/zenodo.10608900](https://zenodo.org/badge/DOI/10.5281/zenodo.10608900.svg)](https://doi.org/10.5281/zenodo.10608900)
  - Train Split (200 tiles from 196 maps)
  - Validation Split (40 tiles from 40 maps)
* [French Cadastre Data Set ![DOI:10.5281/zenodo.10610731](https://zenodo.org/badge/DOI/10.5281/zenodo.10610731.svg)](https://doi.org/10.5281/zenodo.10610731)
  - Train Split (80 tiles from 37 maps)
  - Validation Split (15 tiles from 9 maps)
* [Taiwanese Historical Data Set ![DOI:10.5281/zenodo.14585814](https://zenodo.org/badge/DOI/10.5281/zenodo.14585814.svg)](https://doi.org/10.5281/zenodo.14585814)
  - Train Split (1,478 tiles from 169 maps)
  - Validation Split (166 tiles from 30 maps)
* Competition Test Data
  - [General Rumsey Data Set ![DOI:10.5281/zenodo.10776182](https://zenodo.org/badge/DOI/10.5281/zenodo.10776182.svg)](https://doi.org/10.5281/zenodo.10776182) (700 tiles from 700 maps)
  - French Cadastre Data Set (coming soon)
  - Taiwanese Historical Data Set (coming soon)
* [Sample Data Set ![DOI:10.5281/zenodo.10444912](https://zenodo.org/badge/DOI/10.5281/zenodo.10444912.svg)](https://doi.org/10.5281/zenodo.10444912)
  - 353 tiles from 31 maps of 9 atlases