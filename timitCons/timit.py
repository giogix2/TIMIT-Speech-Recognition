__author__ = 'giova'

import scipy.io
from sklearn import preprocessing
import os.path, sys
import numpy as np
from pylearn2.utils import serial
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.datasets import cache, dense_design_matrix
import pickle

class TIMIT(dense_design_matrix.DenseDesignMatrix):


    def __init__(self, classes_number, which_set):
        self.classes_number = classes_number
        self.path = '/home/gortolan/MachineLearning/'
        self.which_set = which_set
        denseMatrix = pickle.load(open(self.path+self.which_set+'_cons.pkl', "rb" ))
        self.x = denseMatrix.X
        self.y = denseMatrix.y
        super(TIMIT, self).__init__(X=self.x, y=self.y)
