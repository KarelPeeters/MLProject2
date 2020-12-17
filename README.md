# README

## File structure

* `src` the main python source code.
* `glove-rs/src` rust source code for the utility program that creates a cooc matrix from tweets
* `twitter-dataset` the original dataset
* `split-dataset` the dataset after being split up into train and test sets
* `output` intermediate embedding files, trained models and the final submission
* `figures` intermediate data and exported figures for the resport


## Submission reproduction

The code is currently configured to reproduce our final AiCrowd submission, just run `main.py` and the output is written to `output/submission.csv`.

## Main training pipeline
(Optional) Run `split_datasets.py` to create a new train/test split of the dataset. This same split will then be used for all future runs. If not done explicitly this steps happens automatically the first time any code that depends on the dataset runs.

Run `embedding.py` to train a new _GloVe_ embedding. This automatically compiles and runs the intermediate `glove-rs` _Rust_ program to construct the cooc matrix. This program was rewritten from Python because it was too slow. The embedding training can take a while, for the current settings a powerful laptop with GPU takes about half an hour.

Change the bottom of `main.py` to call `main_train` to actually train a model. Different models can be selected by setting the `selected_model` variable. During training a lot of intermediate results are printed to stdout. After training finishes a plot of the loss and accuracy during training is shown, and the model is saved to the `output` folder. Finally `submission_main` is called that loads this model and uses it to create `submission.csv`.

## Other code

* `figures_embedding.py` evaluates the accuracy of a simple model for different embedding sizes.

* `figures_tweets.py` plots some interesting statistics about word frequency and tweet lengths.

* `experiment.py` demonstrates some intuitive structure of the word embedding.