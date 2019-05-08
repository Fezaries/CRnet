# CRnet
An implementation for the paper: Co-Representation Network for Generalized Zero-Shot Learning

Version 1.0

python3.6, pytorch0.4 required

# Dataset
The code uses the ResNet101 features provided by the paper: Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, and follows its GZSL settings. The features can be download here: http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip

# Training

"python CUB_train.py" to train the model;
"python CUB_evaluate.py" to test.

* Loss starts to decrease at approximately episode 30000.
