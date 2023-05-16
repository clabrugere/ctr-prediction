# CTR Prediction

This repository implements some state-of-the-art architectures for CTR prediction, which is about predicting the probability of a click. Because of scale and lantecy constraints, predictions are usually conditionned on tabular data containing sparse and/or dense features. Those class of models try to express interactions of features to alleviate the burden of manual feature engineering, necessary for linear models such as logistic regression for instance.

## Models

| Model     | Year published | Publication                                                                                       |
|-----------|----------------|---------------------------------------------------------------------------------------------------|
| Deep Wide | 2016           | [Wide & Deep for recommender systems](https://arxiv.org/pdf/1606.07792v1.pdf)                     |
| DCN       | 2017           | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)             |
| AutoInt   | 2019           | [Automatic Feature Interaction Learning via Self-Attention](https://arxiv.org/pdf/1810.11921.pdf) |
| DCN-v2    | 2020           | [DCN V2: Improved Deep & Cross Network](https://arxiv.org/pdf/2008.13535v2.pdf)                   |

