
# CSC4025Z: AI Assignment 1

The structure of this directory is as follows:

```script
.
├── mat_to_npy.py            -> Converts original dataset MATLAB files to python .npy files.
├── data_exploration.py      -> Script for exploring the original data.
├── calculate_probs.py       -> Script for calculating some of the conditional probabilities used in the network
├── calculate_thresholds.py  -> Script for calculating some of the thresholds used for testing the network.
├── bayesian_network.py      -> Script for constructing the Bayesian and decision network.
├── raw_to_bin_digits.py     -> Script for converting test data to binary digits for inference.
├── inference.py             -> Script for doing inference.
├── results_plotter.py       -> Scripts for plotting inference results.
├── requirements.txt         -> Python requirements file.
├── network.txt              -> Text representation of the network for making calculating probabilities easier.
└── data/                    -> Contains the original MATLAB files
```

To run the code, please begin by installing the required files (suitable for Windows 10):

```
> pip install -r requirements.txt
```

Then, run the files in the following order to do the following steps:

1. Convert the original MATLAB files to Python .npy files.
1. Calculate the probabilities used in the network.
1. Calcuate the thresholds used for test data.
1. Construct the Bayesian network (there is a flag called `DOING_DECISIONS` which must be set to `True` if you wish to construct the decision network, otherwise it is `False` by default.)
1. Convert the test data to a format that can be passed to the network as evidence.
1. Do inference with the network.
1. Analyse the results.

```
> python mat_to_npy.py
> python calculate_probs.py
> python calculate_thresholds.py
> python bayesian_netork.py
> python raw_to_bin_digits.py
> python inference.py
> python results_plotter.py
```


