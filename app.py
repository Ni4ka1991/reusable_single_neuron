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
weights   = np.random.random( 2 )
biases    = weights.copy()
#biases    = np.zeros( 2 )
#biases    = np.ones( 2 )


#CALCULATIONS

n1 = Neuron( weights[0], biases[0] )
n2 = Neuron( weights[1], biases[1] )

for k in range( 0 , my_data_shape[0]):       #k = number of z axises

    for i in range( 0, my_data_shape[1]):    #i = number of lines
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




E1 = error( experience_data_axis_Z, predicted_data_axis_Z )
L1 = loss( E1 )

system( "clear" )
print( f"z_pred : {predicted_data_axis_Z}" )
print( f"z_exper: {experience_data_axis_Z}" )
print( f"list of errors 1: {E1}" )
print( f"MSE 1: {L1}" )

####

weights   = np.random.random( 2 )
biases    = weights.copy()
n1 = Neuron( weights[0], biases[0] )
n2 = Neuron( weights[1], biases[1] )

for k in range( 0 , my_data_shape[0]):       #k = number of z axises

    for i in range( 0, my_data_shape[1]):    #i = number of lines
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

E2 = error( experience_data_axis_Z, predicted_data_axis_Z )
L2 = loss( E2 )
print( "#"*30 )
print( f"z_pred : {predicted_data_axis_Z}" )
print( f"z_exper: {experience_data_axis_Z}" )
print( f"list of errors2: {E2}" )
print( f"MSE2: {L2}" )















