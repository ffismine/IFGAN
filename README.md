# IFGAN

Code for the paper "Facilitating info-flow within adversarial networks for recommender"

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)



## Usage

###### Install using conda:

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install numpy
```



###### Edit Config in IFGAN.py:

```python
conf = {
    'feature_size': 54,
    'num_epochs': 500,
    'batch_size': 32,
    'dataset_dir' : 'your path here'
}
```



###### Train and Evaluate:

```
python IFGAN.py
```


