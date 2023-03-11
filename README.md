# *INSTINCT* – Inception-based Symbolic Time Intervals Series Classification

## Introduction

This repository is related to the paper "*INSTINCT* – Inception-based Symbolic Time Intervals Series Classification", submitted to the *Information Sciences* journal.

In the paper, we introduce INSTINCT - a novel deep learning based framework for the classification of series of Symbolic Time Intervals (STIs). 

STIs describe events which have a non-zero time duration. 
INSTINCT transforms raw series of STIs data into a two-dimensional representation as real matrices, which are then fed into a novel ensemble of deep CNN models, 
inspired by *Inception-v4* (Szegedy et al. 2017), as well as its recent application in the field of time series classification *InceptionTime* (Fawaz et al. 2020).  

We believe that this repository's contents will contribute to both future research and real-world applications in the field of STIs series classification.

## Repository Contents

The contents of this repository are as follows:
- Source code of INSTINCT
- Code for all the experiments detailed in the paper
- All experimental results' CSVs
- Real-world and synthetic datasets 
- Code for the synthetic datasets generators
- Jupyter Notebook for running the complete flow

## Datasets

- ***Location***: All datasets are available under the [ClassificationDatasets](https://github.com/omerh18/INSTINCT/tree/main/ClassificationDatasets) directory.
- ***Contents***:
    - Real-world benchmark datasets
        - AUSLAN2 (Mörchen and Fradkin 2010)
        - BLOCKS (Mörchen and Fradkin 2010)
        - CONTEXT (Mörchen and Fradkin 2010)
        - HEPATITIS (Patel et al. 2008)
        - PIONEER (Mörchen and Fradkin 2010)
        - SKATING (Mörchen and Fradkin 2010)
    - Synthetic data
        - synthetic data for scalability analysis (Experiment 2 in the paper) 
- ***Format***:
	- *data.txt* - STIs series data 
		- Each line stands for a single STI in a tab separated format, including the series-ID, symbol-type, start-time and finish-time 
	- *labels.txt* - STis series labels
		- Each line specifies the class label of a single STIs series in a tab separated format of the series-ID and the associated class-label

## Code

### INSTINCT

The code of INSTINCT is implemented in [classifier.py](https://github.com/omerh18/INSTINCT/blob/main/classifier.py), and relies on 
[representation_generator.py](https://github.com/omerh18/INSTINCT/blob/main/representation_generator.py) and 
[utils.py](https://github.com/omerh18/INSTINCT/blob/main/utils.py). 

### Experiments

The code for all the experiments detailed in the paper is available under the [experiments](https://github.com/omerh18/INSTINCT/tree/main/experiments) directory.

The Jupyter Notebook [INSTINCT_flow_notebook.ipynb](https://github.com/omerh18/INSTINCT/blob/main/INSTINCT_flow_notebook.ipynb) contains the code for triggering all the experiments, one after the other.

## Experimental Results

All the experimental results' CSVs are available under the [ClassificationResults](https://github.com/omerh18/INSTINCT/tree/main/ClassificationResults) directory.

## Dependencies

- Python 3.8
- Jupyter Notebook
- Packages
    ```
    tensorflow (tensorflow-gpu)
    keras
    scikit-learn
    numpy
    pandas
    tqdm
    ```

## Running Instructions

To run all the experiments, it is recommended to simply run the Jupyter Notebook [INSTINCT_flow_notebook.ipynb](https://github.com/omerh18/INSTINCT/blob/main/INSTINCT_flow_notebook.ipynb).

To run a specific experiment, it is recommended to simply run the cells corresponding to that specific experiment within [INSTINCT_flow_notebook.ipynb](https://github.com/omerh18/INSTINCT/blob/main/INSTINCT_flow_notebook.ipynb).

Note that results will be automatically saved under the [ClassificationResults](https://github.com/omerh18/INSTINCT/tree/main/ClassificationResults) directory. 

Finally, there are also plenty of examples for running INSTINCT under the [experiments](https://github.com/omerh18/INSTINCT/tree/main/experiments) directory, if you like to run INSTINCT in custom settings. 

## References

[1]	Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2020). Inceptiontime: Finding alexnet for time series classification. Data Mining and Knowledge Discovery, 34(6), 1936-1962.

[2] Mörchen, F., & Fradkin, D. (2010, April). Robust mining of time intervals with semi-interval partial order patterns. In Proceedings of the 2010 SIAM international conference on data mining (pp. 315-326). Society for Industrial and Applied Mathematics.

[3]	Patel, D., Hsu, W., & Lee, M. L. (2008, June). Mining relationships among interval-based events for classification. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data (pp. 393-404).

[4]	Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-first AAAI conference on artificial intelligence.
