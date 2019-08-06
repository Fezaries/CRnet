# CRnet
An implementation for the paper: Co-Representation Network for Generalized Zero-Shot Learning

Version 1.01

python3.6, pytorch0.4 required

## Dataset
The code uses the ResNet101 features provided by the paper: Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, and follows its GZSL settings. 

The features can be download here: http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip

## Training

"python CUB_train.py" to train the model.

"python CUB_evaluate.py" to test.

* Loss starts to decrease at approximately episode 30000.

Trained models can be downloaded from https://drive.google.com/open?id=1a1cAG1iOZQAUPoxLG96_xZiW21Him4wS

## Ciation

If you find this work useful for your project, cite:

```
@inproceedings{zhang2019co,
  title={Co-Representation Network for Generalized Zero-Shot Learning},
  author={Zhang, Fei and Shi, Guangming},
  booktitle={International Conference on Machine Learning},
  pages={7434--7443},
  year={2019}
}
```
