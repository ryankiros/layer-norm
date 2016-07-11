# layer-norm
Code and models from the paper "Layer Normalization".

## Dependencies

To use the code you will need:

* Python 2.7
* Theano
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)

## Setup

Available is a file layers.py which contain functions for layer normalization (LN) and 4 RNN layers: GRU, LSTM, GRU+LN and LSTM+LN. The GRU and LSTM functions are added to show what differs from the functions that use LN.

Below we describe how to integrate these functions into existing Github respositories that will allow you to perform the same experients as in the paper. We also make available the trained models that we used to compute curves and numbers in the paper.

NOTE: it is highly encouraged to use CNMeM when using layer norm. Just add cnmem = 1 to your Theano flags.

## Order-embeddings

The order-embeddings experiments make use of the respository from Ivan Vendrov et al available [here](https://github.com/ivendrov/order-embedding). To train order-embeddings with layer normalization:

* Clone the above repository
* Add the *lngru_layer* and *param_init_lngru* functions to layers.py in the order-embeddings repo
* Add 'lngru': ('param_init_lngru', 'lngru_layer'), to layers
* In driver.py, replace 'encoder': 'gru' with 'encoder': 'lngru'
* Follow the instructons on the main page to train a model

Available below is a download to the model used to report results in the paper:

    wget http://www.cs.toronto.edu/~rkiros/lngru_order_add0.npz
    wget http://www.cs.toronto.edu/~rkiros/lngru_order_add0.pkl

Once downloaded, follow the instructions on the main page for evaluating models. This will allow you to reproduce the numbers reported in the table for order-embeddings+LN.

## Skip-thoughts

The skip-thoughts experiments make use of the repository from Jamie Ryan Kiros et al available [here](https://github.com/ryankiros/skip-thoughts). To train skip-thoughts with layer normalization:

* Clone the above repository
* Add the *lngru_layer* and *param_init_lngru* functions to layers.py in training/layers.py in the skip-thoughts repo
* Add 'lngru': ('param_init_lngru', 'lngru_layer'), to layers
* In training/train.py, replace encoder='gru' with encoder='lngru' and replace decoder='gru' with decoder='lngru'
* Follow the instructions in the training directory to train a model

Below is the skip-thoughts model trained for 1 month using layer normalization:

    wget http://add-model-here

Once downloaded, follow Step 4 in the training directory to load the model. This model will allow you to reproduce the reported results in the last row of the table. Step 5 describes how to use the model to encode new sentences into vectors.

## Attentive-reader

The attentive reader experiment makes use of the repository from Tim Cooijmans et al [here](https://github.com/cooijmanstim/Attentive_reader/tree/bn). To train an attentive reader model:

* Clone the above repository and obtain the data (more details to follow)
* Add the *lnlstm_layer* and *param_init_lnlstm* functions to layers.py in codes/att_reader/layers.py
* Add 'lnlstm': ('param_init_lnlstm', 'lnlstm_layer'), to layers
* Follow the instructions for training a new model and replace the argument --unit_type lstm with --unit_type lnlstm

Below are the log files from the model trained using layer normalization:

    wget http://add-model-here

These can be used to reproduce the layer norm curve from the paper.
