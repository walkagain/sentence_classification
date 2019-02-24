**[most code of this project comes from "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

A bi-direction RNN class was added to the original code, so you can decide train your model with different neural network model.  

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Run

Print parameters:

```bash
./main.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --action: specify running mode [train, eval]
    (default: 'train')
  --[no]allow_soft_placement: Allow device soft device placement
    (default: 'true')
  --attn_size: size of attention layer in rnn
    (default: '256')
    (an integer)
  --checkpoint_dir: Checkpoint directory from training run
    (default: '')
  --checkpoint_every: Save model after this many steps
    (default: '100')
    (an integer)
  --dev_sample_percentage: Percentage of the training data to use for validation
    (default: '0.1')
    (a number)
  --dropout_keep_prob: Dropout keep probability
    (default: '0.5')
    (a number)
  --embedding_dim: Dimensionality of character embedding
    (default: '128')
    (an integer)
  --eval_batch_size: Batch Size For evaluating
    (default: '64')
    (an integer)
  --[no]eval_train: Evaluate on all training data
    (default: 'false')
  --evaluate_every: Evaluate model on dev set after this many steps
    (default: '100')
    (an integer)
  --filter_sizes: Comma-separated filter sizes
    (default: '3,4,5')
  --l2_reg_lambda: L2 regularization lambda
    (default: '0.0')
    (a number)
  --[no]log_device_placement: Log placement of ops on devices
    (default: 'false')
  --lstm_size: size of hidden layer for rnn model
    (default: '128')
    (an integer)
  --model: specify one model[cnn, rnn] to train
    (default: 'cnn')
  --negative_data_file: Data source for the negative data.
    (default: './data/rt-polaritydata/rt-polarity.neg')
  --num_checkpoints: Number of checkpoints to store
    (default: '5')
    (an integer)
  --num_epochs: Number of training epochs
    (default: '200')
    (an integer)
  --num_filters: Number of filters per filter size
    (default: '128')
    (an integer)
  --num_layers: hidden layers for rnn
    (default: '2')
    (an integer)
  --positive_data_file: Data source for the positive data.
    (default: './data/rt-polaritydata/rt-polarity.pos')
  --train_batch_size: Batch Size For training
    (default: '64')
    (an integer)

```

Train:

```bash
./main.py --action train <--model rnn>
```

Evaluate:

```bash
./main.py --action eval --eval_train --checkpoint_dir="./runs/1550735612/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `evaluate() function in eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
