import numpy as np
from os import system
from neuron import *
#from data1 import *
from data import *

predicted_data = experience_data.copy()
my_data_shape = predicted_data.shape #(nr of layers, nr of lines, nr of columns )

def mse_calculation( array_shape, w, b ):

    n1 = Neuron( w[0], b[0] )
    n2 = Neuron( w[1], b[1] )

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
