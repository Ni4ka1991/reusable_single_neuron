#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
#from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax  = plt.axes( projection = '3d' )

#all_data
experience = [ 5,  10,  15,    5,  10, 15,    2,  4,   6   ] #years
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60,  70  ] #k$/year

#arrays
#[ x    2*x        3*x   ]
#[ x    2*x        3*x   ]
#[ x   x + k     x + 2*k ]

#[ 5,   10,  y1 ] #15
#[ 100, 200, y2 ] #300
#[ 120, 150, y3 ] #180


# NEURON LOGIC ######

class Neuron:
    def __init__( self, w = 0, b = 0 ):
        self.w = w
        self.b = b
    
    def forward( self, X ):
        return [ ( self.w * x + self.b ) for x in X  ] 


def error( Y, Y_pred ):
    return [ ( y - y_pred ) for y, y_pred in zip( Y, Y_pred )  ]


def loss( E ):                                     
    return sum( [ e * e for e in E ] ) / len( E )                      #MSE

def accuracy( E ):
    return sum( [ abs( e ) for e in E ] ) / len( E )

# ************ ######

experience_data = np.array( [ 
               [  #axis z = 0
#                 x   y    z     = data! not axis
                 [5,  10,  15 ], #0
                 [50, 100, 150], #1
                 [60, 70,  80 ]  #2
               ],
               
               [ #axis z = 1
                 [5,  10, 15],
                 [25, 50, 75],
                 [30, 35, 40]
               ],
               
               [ #axis z = 2
                 [2,  4,   6  ],
                 [50, 100, 150],
                 [50, 60,  70 ]
               ]
                    ] )

predicted_data = experience_data.copy()

weights = np.zeros( 2 )
biases    = weights.copy()





test_data = np.array( [ 8, 10, 14, 5 ] )

n = Neuron( weights[0], biases[0] )
a = n.forward( test_data )
print(a)
input( "hit enter ..." )

predicted_data[0][0][2] = 88
predicted_data[0][1][2] = 88
predicted_data[0][2][2] = 88


predicted_data[1][0][2] = 88
predicted_data[1][1][2] = 88
predicted_data[1][2][2] = 88


predicted_data[2][0][2] = 88
predicted_data[2][1][2] = 88
predicted_data[2][2][2] = 88

system( "clear" )
print(predicted_data)





































