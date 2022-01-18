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
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60, 70   ] #k$/year


#arrays
#[ x    2*x        3*x   ]
#[ x    2*x        3*x   ]
#[ x   x + k     x + 2*k ]

#[ 5,   10,  y1 ] #15
#[ 100, 200, y2 ] #300
#[ 120, 150, y3 ] #180


#calculation of y1 = f(x1); y2 = f(x2) => f identical =>>>>> 

#[ x,   2 * x,   3 * x   ]
#[ x,  x + k,  x + 2 * k ]


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

d1 = np.array( [ [5, 10, 15], [50, 100, 150], [60, 70, 80] ] )
d2 = np.array( [ [5, 10, 15], [25, 50,  75 ], [30, 35, 40] ] )
d3 = np.array( [ [2, 4,  6 ], [50, 100, 150], [50, 60, 70] ] )
d_total = ( [ 
               [ [5, 10, 15], [50, 100, 150], [60, 70, 80] ],
               [ [5, 10, 15], [25, 50,  75 ], [30, 35, 40] ],
               [ [2, 4,  6 ], [50, 100, 150], [50, 60, 70] ]
          ] )



a = np.ones(3)
a2 = np.ones(( 3, 2 ))

b = np.zeros(3)
b2 = np.zeros(( 3, 2 ))
c = np.random.random(3)
std_c = c.std()
mean_c = c.mean()
system( "clear" )
print( a )
print( a2 )
print( b )
print( b2 )
print( "#"*20 + f"\nResulting matrix:\n{d_total}\n" + "#"*20 )
print( f"Random array c 3*1 = {c}" )
print( f"Standard deviation c: {std_c}" )
print( f"Arithmetic mean c: {mean_c}" )
