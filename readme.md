# CTR Prediction

This repository implements some state-of-the-art architectures for CTR prediction, which is about predicting the probability of a click.

Because of scale and lantecy constraints, predictions are usually conditionned on tabular data containing sparse and/or dense features, both categorical and continuous. Those class of models try to express high order interactions of features to alleviate the burden of manual feature engineering, necessary for linear models such as logistic regression for instance.

## Models

| Model     | Year published | Publication                                                                                                                      |
| --------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Deep Wide | 2016           | [Wide & Deep for recommender systems](https://arxiv.org/pdf/1606.07792v1.pdf)                                                    |
| DCN       | 2017           | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)                                            |
| AutoInt   | 2019           | [Automatic Feature Interaction Learning via Self-Attention](https://arxiv.org/pdf/1810.11921.pdf)                                |
| DCN-v2    | 2020           | [DCN V2: Improved Deep & Cross Network](https://arxiv.org/pdf/2008.13535v2.pdf)                                                  |
| FinalMLP  | 2023           | [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/pdf/2304.00902v3.pdf)                          |
| GDCN      | 2023           | [Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction](https://dl.acm.org/doi/pdf/10.1145/3583780.3615089) |

## Dependencies

Thie repository has the following dependencies:

- python 3.9+
- pytorch 2.0+ (only for pytorch implementations)
- tensorflow 2.12+ (only for tensorflow implementations)

## Usage

Copy the implementation of the model you want or clone the repository. Then simply train as usual.

For example with Tensorflow:

```python
# load your tensorflow dataset
train_data = ...
val_data = ...

model = FinalMLP(
   dim_input=num_features,
   num_embedding=num_embedding,
   dim_embedding=32,
   dropout=0.2,
)

# train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss="binary_crossentropy")
model.fit(
   train_data,
   validation_data=val_data,
   epochs=20,
)

# make predictions
y_pred = model.predict(X_test)

```
