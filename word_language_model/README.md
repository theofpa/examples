# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py with model_type=RNN_TANH lr=0.2 nhid=150 nlayers=1 bptt=20 dropout=0.5 save=rnn_01.pt emsize=250 tied=False epochs=3
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
$ python main.py print_config
INFO - main - Running command 'print_config'
INFO - main - Started
Configuration (modified, added, typechanged, doc):
  batch_size = 32                    # batch size
  bptt = 35                          # sequence length
  clip = 0.25                        # gradient clipping
  dataset = './data'                 # location of the data corpus
  dropout = 0.2                      # dropout applied to layers (0 = no dropout)
  emsize = 200                       # size of word embeddings
  epochs = 40                        # upper epoch limit
  log_interval = 200                 # report interval
  lr = 20.0                          # initial learning rate
  model_type = 'LSTM'                # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  nhid = 200                         # number of hidden units per layer
  nlayers = 2                        # number of layers
  save = 'model.pt'                  # path to save the final model
  seed = 257                         # the random seed for this experiment
  tied = True                        # tie the word embedding and softmax weights
INFO - main - Completed after 0:00:00
```

Perplexities on PTB are equal or better than
[Recurrent Neural Network Regularization (Zaremba et al. 2014)](https://arxiv.org/pdf/1409.2329.pdf)
and are similar to [Using the Output Embedding to Improve Language Models (Press & Wolf 2016](https://arxiv.org/abs/1608.05859) and [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf), though both of these papers have improved perplexities by using a form of recurrent dropout [(variational dropout)](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks).
