#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
from neuron import *
#from data1 import *
from data import *
from mse_calculation import *


#PREDICTED DATA 
my_data_shape = predicted_data.shape #(nr of layers, nr of lines, nr of columns )

#biases initialization
weights   = np.zeros( 2 )
#weights   = np.ones( 2 )
#weights   = np.random.random( 2 )
#biases    = weights.copy()
biases    = np.zeros( 2 )
#biases    = np.ones( 2 )


#CALCULATIONS

def find_mse( weights, biases ):
    weights   = np.random.random( 2 )
    biases    = np.random.random( 2 )
    mse = mse_calculation( my_data_shape, weights, biases )
    return mse

delta = abs( find_mse( weights, biases ) - find_mse( weights, biases ))

print( delta )

