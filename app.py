#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
from neuron import *
#from data1 import *
from data import *

#PREDICTED DATA 
predicted_data = experience_data.copy()
my_data_shape = predicted_data.shape #(nr of layers, nr of lines, nr of columns )

#biases initialization
#weights   = np.zeros( 2 )
#weights   = np.ones( 2 )
#weights   = np.random.random( 2 )
#biases    = weights.copy()
#biases    = np.zeros( 2 )
#biases    = np.ones( 2 )


#CALCULATIONS

weights   = np.random.random( 2 )
biases    = np.random.random( 2 )

def mse_calculation( array_shape, ):

    n1 = Neuron( weights[0], biases[0] )
    n2 = Neuron( weights[1], biases[1] )

    for k in range( 0 , array_shape[0]):       #k = number of z axises

        for i in range( 0, array_shape[1]):    #i = number of lines
            predicted_data[ k, i, 2 ] = np.array( n1.forward( predicted_data[ k, :, 0 ] ))[i] + np.array( n2.forward( predicted_data[ k, :, 1 ] ))[i]
            i += 1
        k += 1


    experience_data_axis_Z = np.hstack(( experience_data[ 0, :, 2 ],\
                                         experience_data[ 1, :, 2 ],\
                                         experience_data[ 2, :, 2 ]\
                                         ))

    predicted_data_axis_Z  = np.hstack(( predicted_data[ 0, :, 2 ], \
                                         predicted_data[ 1, :, 2 ],\
                                         predicted_data[ 2, :, 2 ]\
                                         ))


    E = error( experience_data_axis_Z, predicted_data_axis_Z )
    L = loss( E )
    return L


mse1 = mse_calculation( my_data_shape )
weights   = np.random.random( 2 )
biases    = np.random.random( 2 )
mse2 = mse_calculation( my_data_shape )
delta = mse1 - mse2

print( delta )

