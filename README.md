# Amazon Sentiment Review

## Getting Started
This notebook elaborates how to dockerize a data science project using AWS services and Docker.
Before starting, make sure that your IAM has the following permissions:
- AmazonS3FullAcess
- AmazonSageMakerFullAccess
- AWSGlueConsoleSageMakerNotebookFullAccess
- AWSImageBuilderFullAccess
- AmazonSageMakerPipelinesIntegrations
- AmazonSageMakerGroundTruthExecution

## Sagemaker Studio Lab
The following notebook can be run on your local machine or preferably inside a sagemaker studio lab notebook.

## Fetching the Data

The data is available for free on Kaggle:
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews

It's highly recommended to start out small with a subset of the data as opposed to the whole dataset to avoid excessive AWS charges.

## Running the Workflow

Most of the work is done inside the [Preprocessing-with-container.ipynb](https://github.com/NadimKawwa/amazon-review-docker/blob/main/Preprocessing-with-container.ipynb) notebook. In addition, the python scripts are written via the notebook.

#### Trouble Loading the Notebook?
If the notebook takes too long to load, this website can help:
https://nbviewer.org/
