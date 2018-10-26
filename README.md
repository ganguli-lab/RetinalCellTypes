# RetinalCellTypes

This repository provides code accompanying the paper "The emergence of multiple retinal cell types through
efficient coding of natural movies," to appear at NIPS 2018.

Description of code:

STEPHANE PUT YOUR DESCRIPTIONS HERE

NeuralNetSimulation.py contains the code needed to train the nonlinear neural network models referred to in the paper.   It runs on a GPU using Keras with Tensorflow backend.  The code runs properly using Python 3.X, Tensorflow vXX, Cuda vXX, and Keras vXX (other versions may work as well but there is no guarantee). It performs a grid search over possible cell type allocations (number of cells for each type), noise values, and l1 regularization values.  Stores the results in a numpy array which can be processed using Jupyter notebook AnalyzeNeuralNetSimulation.ipynb.

AnalyzeNeuralNetSimulation.ipynb contains code used to plot all neural network-related results in the paper.  Comments in the code describe the role of each part refer to particular figures in the paper.
