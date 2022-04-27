# Coding : LSTM vs. GRU
LSTM and GRU are two popular models designed to prevent vanishing gradient problem. GRUs are simpler
than LSTMs, which control the flow of information like the LSTM unit. From empirical experience, training a
GRU is faster than that of LSTM. However, LSTM and its variants outperform GRUs if the training data are
sufficient. More details can be found in [here](https://arxiv.org/abs/1702.01923). In this question, you will first implement an LSTM
and a GRU, and then evaluate them on a small dataset for sentiment analysis: IMDB5.

## Q5 : Instructions

1. Prepare your environment. Download the code here and follow the instructions below.
    - Prepare your python3 environment, and install PyTorch 1.8.06
    - install torchtext (`conda install torchtext=0.9.0 -c pytorch`).
    - install tokenizer (`conda install spacy=3.2 -c conda-forge and python -m spacy download en core web sm`).
1. Complete the code of GRUCell and LSTMCell in the file assignment code.py. The missing parts are
highlighted. Please don’t change other parts of the code, otherwise TAs may not be able to run the code
in your submission.
   - Read the comments in the code first. Cells are the backbone of recurrent neural networks. A recurrent layer contains a cell object. The cell implements the computations of one single step, while the recurrent layer calls the cell and performs recurrent computations (with loop), as illustrated during exercise session 3.
   - Complete the code in classes `GRUCell_assignment` and `LSTMCell_assignment`. Each class contains two functions: initialization `__init__()` and forward `forward()`. Below are the definitions of the two cells. You can also refer to the PyTorch documentation.