# Instructions Question 4

- Download the [code](https://polybox.ethz.ch/index.php/s/pSN67mjEfSjRKtQ) and follow the instructions :
  - Install Anaconda3 or Miniconda4 and create a virtual environment by running the following commands in the project directory: 
    - `conda env create -f env.yml`
    - `conda activate a1`
  - We will start with `word2vegenc.py`. Go ahead and fill out the code for for `negSamplingLossAndGradient`,
  `Skip-gram`. You will need to calculate the loss and the gradients.
  - Test out the gradients by running `python word2vec.py`
  - Now let’s run FastText for 20000 iterations, do this by python run.py (this will take around 3-4 hours, so please plan accordingly)
  - The code will generate a plot, paste that in your assignment, along with the plot submit all the python files (*.py) in a zip file.
- What do you think are the advantages a subword approach such as fastText has over a character embedding-based approach? 
- Can you think of a few examples where subword information might hurt the embedding’s performance?