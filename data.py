import numpy as np

#all_data
experience = [ 5,  10,  15,    5,  10, 15,    2,  4,   6   ] #years
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60,  70  ] #k$/year

experience_data = np.array( [ 
               [  #axis z = 0
#                 x   y    z     = data! not axis
                 [5,  10,  15 ], #0
                 [50, 100, 150], #1
                 [60, 70,  80 ]  #2
               #  0   1    2
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





