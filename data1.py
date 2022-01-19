import numpy as np

#all_data
experience = [ 5,  10,  15,    5,  10, 15,    2,  4,   6   ] #years
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60,  70  ] #k$/year

experience_data = np.array( [ 
               [  #axis z = 0
#                 x   y    z     = data! not axis
                 [5,  10,  15,  20, 48 ], #0
                 [5,  10,  15,  4,  12 ], #0
                 [50, 100, 150, 56, 10 ], #1
                 [60, 70,  80,  8,  14 ]  #2
               #  0   1    2
               ],
               
               [ #axis z = 1
                 [5,  10, 15, 16, 74 ],
                 [5,  10, 15, 56, 13 ], #0
                 [25, 50, 75, 66, 77 ],
                 [30, 35, 40, 45, 15 ]
               ],
               
               [ #axis z = 2
                 [2,  4,   6,  14, 35 ],
                 [5,  10,  15, 56, 98 ], #0
                 [50, 100, 150, 33, 22],
                 [50, 60,  70, 65, 87 ]
               ]
                    ] )




