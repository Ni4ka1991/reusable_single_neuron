#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
from neuron import *
from data import *



predicted_data = experience_data.copy()

#biases initialization
#weights   = np.zeros( 2 )
weights   = np.ones( 2 )
biases    = weights.copy()


#PREDICTED DATA 

n1 = Neuron( weights[0], biases[0] )
n2 = Neuron( weights[1], biases[1] )


for k in range( 0 , 3 ):
    for i in range( 0, 3 ):
        predicted_data[ k, i, 2 ] = np.array( n1.forward( predicted_data[ k, :, 0 ] ))[i] + np.array( n2.forward( predicted_data[ k, :, 1 ] ))[i]
        i += 1
    k += 1



#predicted_data[ k, i, 2 ] = np.array( n1.forward( predicted_data[ k, :, 0 ] ))[i] + np.array( n2.forward( predicted_data[ k, :, 1 ] ))[i]



system( "clear" )
print(predicted_data)

















