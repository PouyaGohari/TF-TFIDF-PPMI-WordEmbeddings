# TF-TFIDF-PPMI-WordEmbeddings
This repository contains the solutions to the second computer assignment for the Natural Language Processing course at the University of Tehran. The assignment focuses on various text processing methods, feature extraction techniques, and their application in sentiment analysis,  sarcasm detection, and word embedding training.


## Table of Contents

- [Introduction](#introduction)
- [Assignment Overview](#assignment-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [Report](#report)
- [License](#license)

## Introduction

This project explores multiple NLP techniques, including term frequency-based methods, sentiment analysis with logistic regression, and training word embeddings. Each task involves different methods of text representation and analysis, aimed at understanding and applying various NLP techniques in practical scenarios.

## Assignment Overview

### Problem 1: Term Frequency Methods and Naive Bayes

- Implement and compare three text processing methods: Term Frequency (TF), Term Frequency-Inverse Document Frequency (TF-IDF), and Positive Pointwise Mutual Information (PPMI).
- Train a Naive Bayes classifier on the datasets processed using these methods.
- Analyze and compare the performance of these methods.

**Dataset:** The dataset for this problem can be downloaded from [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).

### Problem 2: Sarcasm Detection using Logistic Regression and GloVe

- Use pre-trained GloVe embeddings to represent text data.
- Train a logistic regression model for sarcasm detection.
- Evaluate the model's performance on a sarcasm detection dataset.

**Pre-trained Embeddings:** Download the GloVe embeddings from [GloVe 6B](https://nlp.stanford.edu/data/glove.6B.zip).

**Dataset Path:** The dataset for sarcasm detection is located at [`sarcasm.json`](sarcasm.json).

### Problem 3: Word2Vec Training using Skipgram with Negative Sampling

- Train a Word2Vec model using the Skipgram approach with negative sampling.
- Use a text corpus from one of Sherlock Holmes' stories.
- Analyze the learned word embeddings.

**Dataset Path:** The text corpus for training the Word2Vec model is located at [`advs.txt`](advs.txt).



## Prerequisites

Before you begin, ensure you have met the following requirements:

- Required Python packages: `numpy`, `scikit-learn`, `nltk`, `pandas`.
- Familiarity with basic NLP concepts, word embeddings, and machine learning models.

## Installation

To clone and run this repository locally:
```sh
git clone https://github.com/PouyaGohari/TF-TFIDF-PPMI-WordEmbeddings.git
cd TF-TFIDF-PPMI-WordEmbeddings
```

## Usage

The tasks are implemented in three separate Jupyter notebooks, each corresponding to a different question in the assignment.

1. **Question 1: Term Frequency Methods and Naive Bayes**
   - Open the notebook located at [`Q1.ipynb`](Q1.ipynb).
   - Follow the sections within the notebook to:
     - Implement and analyze TF, TF-IDF, and PPMI methods.
     - Train and evaluate a Naive Bayes classifier using these methods.

2. **Question 2: Sarcasm Detection using Logistic Regression and GloVe**
   - Open the notebook located at [`Q2.ipynb`](Q2.ipynb).
   - Follow the sections within the notebook to:
     - Train and evaluate a logistic regression model for sarcasm detection using GloVe embeddings.

3. **Question 3: Word2Vec Training using Skipgram with Negative Sampling**
   - Open the notebook located at [`Q3.ipynb`](Q3_v2.ipynb).
   - Follow the sections within the notebook to:
     - Train a Word2Vec model on a Sherlock Holmes text corpus using Skipgram with negative sampling.
     - Analyze the learned word embeddings.

Each notebook is organized sequentially, allowing you to work through the specific problem in order.


## Results and Analysis

- **Term Frequency Methods:** The performance of TF, TF-IDF, and PPMI when combined with Naive Bayes is compared and discussed.
- **Sarcasm Detection:** The results of sarcasm detection using logistic regression with GloVe embeddings are analyzed.
- **Word2Vec Embeddings:** The quality and characteristics of the word embeddings trained on the Sherlock Holmes corpus are evaluated.

## Report

A comprehensive report detailing the methodology, implementation, results, and analysis for each task is available [here](Report/Report.pdf).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
