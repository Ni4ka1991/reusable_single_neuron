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

def mse_calculation( array_shape, ):
    weights   = np.random.random( 2 )
    biases    = weights.copy()

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

    print( f"z_pred : {predicted_data_axis_Z}" )
    print( f"z_exper: {experience_data_axis_Z}" )
    print( f"list of errors : {E}" )
    print( f"MSE: {L}" )

system( "clear" )
mse_calculation( my_data_shape )
mse_calculation( my_data_shape )

