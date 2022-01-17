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
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number   y1 = 10*x, y2 = 5*x, y3 = 25*x
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60, 70   ] #k$/year  z = f ( x, y )



#projects in one month   = [ 4.1,  8.33, 12.5,
#                            2.08, 4.1,  6.25,                               ]
#                            4.1,  8.33, 12.5 ]




#    5                 10                  15               years exp

# 60   30            70    35           80    40
# -- , --- = 1.2     ---,  --- = 0.7    ---,  --- = 0.5(3)  k$/one project  experiense ^ , earn v
# 50   25            100   50           150   75


#    2        4          6                                  years exp

# 50          60         70
# -- = 1   --- = 0.6     --- = 0.466666                     k$/one project  experiense ^ , earn v
# 50         100         150


#experience_data  = [ 2, 4,   5,   6,    10,  15   ]
#earn_one_project = [ 1, 0.6, 1.2, 0.47, 0.7, 0.53 ]


#ax.plot3D( experience, projects, earn, 'black', label ='parametric curve' )
#ax.set_title( 'All data sets' )

#plt.plot( experience, earn, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()

#data for testing
#experience = [ 5,   10  ] #years
#projects   = [ 100, 200 ] #number
#earn       = [ 120, 150 ] #k$/year


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


# CALCULATIONS ######

edl = 20                 #experience_data_lenght = 20

## random data list ###
def produce_random_data_list( len_list = 20 ):
    D = []
    for i in range( len_list ):
        d = np.random.normal()
        D.append( d )
    return D

W = produce_random_data_list( edl )    # list of 20 random weights
B = produce_random_data_list( edl )    # list of 20 random biases


## function for calculate a rough approximation loss

def rough_approximation_loss( X, Y, W, B ):
    L = []
    for i in range( len( W )):
        n = Neuron( W[i], B[i] )
        Y_predicted = n.forward( X )
        E = error( Y, Y_predicted )
        L.append( loss( E ))
    return L


L_W = rough_approximation_loss( experience, projects, W,             B = [0] * edl )
L_B = rough_approximation_loss( experience, projects, W = [0] * edl, B             )





def search_least_loss_coefficients( X, Y, W, B, k ):
    LL = []
    for i in range( k ):
        LL.append( rough_approximation( X, Y, W, B )



#L_B = rough_approximation( B,  experience, projects )




#n = Neuron( 0, 0 )                                       
#Y_predicted = n.forward( experience )     #calculation for all X, Y_predicted ( w = 3, b = 5 )

#E = error( projects, Y_predicted )
#L = loss( E )

system( "clear" )
print( f"W_list = {W}" )
print( f"B_list = {B}" )
print( f"L(W) = {L_W}" )







