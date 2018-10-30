# RetinalCellTypes

This repository provides code accompanying the paper "The emergence of multiple retinal cell types through
efficient coding of natural movies," to appear at NIPS 2018.

# Code for linear simulation

C1_SPEEDY_WF.pynb: Reconstruction error with respect to RMS firing rate budget for one or two types (fig. 6c)

C3_RATIO_SLIDER_VAR.ipynb: Reconstruction error as a function of the fraction of 'midget' cells at fixed RMS firing rate budget (fig 3d)

C4_RATIO_SLIDER_SPIKES.ipynb: RMS Firing rate busdget as a function of the fraction of 'midget' cells at fixed reconstruction error (fig 3e)


# Neural Network simulation



NeuralNetSimulation.py contains the code needed to train the nonlinear neural network models referred to in the paper.   It runs on a GPU using Keras with Tensorflow backend.  The code runs properly using Python 3.X, Tensorflow vXX, Cuda vXX, and Keras vXX (other versions may work as well but there is no guarantee). It performs a grid search over possible cell type allocations (number of cells for each type), noise values, and l1 regularization values.  Stores the results in a numpy array which can be processed using Jupyter notebook AnalyzeNeuralNetSimulation.ipynb.

AnalyzeNeuralNetSimulation.ipynb contains code used to plot all neural network-related results in the paper.  Comments in the code describe the role of each part refer to particular figures in the paper.
