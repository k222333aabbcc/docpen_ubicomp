# DocPen_UbiComp

This is a Pytorch implementation of DocPen test project, providing the code, weights, and test sets needed for model testing.

## Overview

This project is used to test and evaluate the performance of the DocPen model. This project contains the resources needed to run model tests, including code, pre-trained weights, and a test dataset. Since the full dataset is too large, we only provide the test dataset for quick evaluation.

### File Structure

- `model.py`: The main test script.
- `weights/`: The folder containing the model weights.
- `data/`: The folder containing the test dataset.
- `utils/`: Auxiliary tools and functions.
- `SA_ConvLSTM_Pytorch/`: Implementation of ConvLstm, adjusted from [SA-ConvLSTM-Pytorch](https://github.com/tsugumi-sys/SA-ConvLSTM-Pytorch.git) .

## Dataset

Since the complete dataset is too large, we only provide the test dataset. The dataset is divided as follows:

- **Proportional division**: Each author's data is divided into training data and test data in a ratio of 4:1.
- **Document independence**: The samples in the training set and the test set come from different documents to ensure the fairness of the model evaluation.

You need to unzip the dataset before using it.

## Examples

```bash
python model.py
```

**model.py** will load the model, evaluate using the test dataset, and print the results.
