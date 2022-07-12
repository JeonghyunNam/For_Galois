from cmath import nan
from json import load

from sympy import Symbol, factor_list
from sympy import factor
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib
import csv
from numpy import genfromtxt
import pandas as pd

def makePoly(prepoly):
    a,b,c,d,e,f = [prepoly[i] for i in range(6)]
    if f < 2:
        f+=1
    elif f == 2:    
        e,f = e+1, 0
    if e > 2:
        d,e,f = d+1, 0, 0
    if d > 2:
        c,d,e,f = c+1, 0, 0, 0
    if c > 2:
        b,c,d,e,f = b+1, 0, 0, 0, 0
    if  b > 2:
        a,b,c,d,e,f = a+1,0,0,0,0,0

    if a>2:
        return True, (a,b,c,d,e,f)
    else:
        return False, (a,b,c,d,e,f)

def transPoly(poly):
    assert len(poly) == 6
    p_trans = [0]*4

    a, b, c, d, e ,f = [poly[i] for i in range(6)]
    p_trans[0] = (5*a*c - 2*b*b)/(5*a*a)                                                                            # third
    p_trans[1] = (25*a*a*d - 15*a*b*c+4*b*b*b)/(25*a*a*a)                                                           # second 
    p_trans[2] = (125*a*a*a*e - 50*a*a*b*d + 15*a*b*b*c - 3*b*b*b*b)/(125*a*a*a*a)                                  # first
    p_trans[3] = (3125*a*a*a*a*f - 625*a*a*a*b*e + 125*a*a*b*b*d - 25*a*b*b*b*c + 4*b*b*b*b*b)/(3125*a*a*a*a*a)     # const
    
    return p_trans

def ProbeniusPoly(poly):
    assert len(poly) == 4
    p_proben = [1]+[0]*6

    p, q, r, s = [poly[i] for i in range(4)]
    p_proben[1] = 8*r                                                                                       # fifth
    p_proben[2] = 2*p*q - 6*p*p*r + 40*r*r -50*q*s                                                          # fourth
    p_proben[3] = (-2*q*q*q*q + 21*p*q*q*r -40*p*p*r*r +160*r*r*r                                           
                     -15*p*p*q*s -400*q*r*s +125*p*s*s)                                                     # third
    p_proben[4] = (p*p*q*q*q*q - 6*p*p*p*q*q*r - 8*q*q*q*q*r + 9*p*p*p*p*r*r + 76*p*q*q*r*r
                    -136*p*p*r*r*r + 400*r*r*r*r - 50*p*q*q*q*s + 90*p*p*q*r*s
                    -1400*q*r*r*s +625*q*q*s*s +500*p*r*s*s)                                                # second 
    p_proben[5] = (-2*p*q*q*q*q*q*q + 19*p*p*q*q*q*q*r - 51*p*p*p*q*q*r*r + 3*q*q*q*q*r*r   
                     +32*p*p*p*p*r*r*r + 76*p*q*q*r*r*r + 256*p*p*r*r*r*r + 512*r*r*r*r*r 
                     -31*p*p*p*q*q*q*s - 58*q*q*q*q*q*s + 117*p*p*p*p*q*r*s + 105*p*q*q*q*r*s
                     +260*p*p*q*r*r*s - 2400*q*r*r*r*s - 108*p*p*p*p*p*s*s - 325*p*p*q*q*s*s 
                     +525*p*p*p*r*s*s + 2750*q*q*r*s*s - 500*p*r*r*s*s + 625*p*q*s*s*s -3125*s*s*s*s)       # first

    p_proben[6] = (q*q*q*q*q*q*q*q - 13*p*q*q*q*q*q*q*r + p*p*p*p*p*q*q*r*r + 65*p*p*q*q*q*q*r*r 
                     -4*p*p*p*p*p*r*r*r -128*p*p*p*q*q*r*r*r +17*q*q*q*q*r*r*r +48*p*p*p*p*r*r*r*r
                     -16*p*q*q*r*r*r*r - 192*p*p*r*r*r*r*r + 256*r*r*r*r*r*r -4*p*p*p*p*p*q*q*q*s
                     -12*p*p*q*q*q*q*q*s +18*p*p*p*p*p*p*q*r*s +12*p*p*p*q*q*q*r*s -124*q*q*q*q*q*r*s
                     +196*p*p*p*p*q*r*r*s +590*p*q*q*q*r*r*s -160*p*p*q*r*r*r*s -1600*q*r*r*r*r*r*r*s
                     -27*p*p*p*p*p*p*p*s*s -150*p*p*p*p*q*q*s*s -125*p*q*q*q*q*s*s -99*p*p*p*p*p*r*s*s 
                     -725*p*p*q*q*r*s*s + 1200*p*p*p*r*r*s*s +3250*q*q*r*r*s*s -2000*p*r*r*r*s*s
                     -1250*p*q*r*s*s*s +3125*p*p*s*s*s*s -9375*r*s*s*s*s)                                   # const

    return p_proben

def deterSolvability(poly):
    # 1. Determine irreducibility
    x = Symbol('x')
    inputPoly = 0
    for i in range(6):
       inputPoly += poly[i]*x**(5-i)
    factorLength = len(factor_list(inputPoly)[1])

    if (factorLength != 1 or poly[0] == 0):
        return True
    
    # 2. Use Theorem for irreducible polynomial
    new_poly = transPoly(poly)
    proben_poly = ProbeniusPoly(new_poly)
    # print(f"translated : {new_poly}")
    # print(f"probenious : {proben_poly}")
    
    f20 = 0
    for i in range(7):
       f20 += proben_poly[i]*x**(6-i)
    # print(f"factored : {f20}")
    f20_factorLength = len(factor_list(f20)[1])

    if f20_factorLength == 1:
        return False
    else:
        return True

def makeData():
    prepoly = (-2,-2,-2,-2,-2,-2)
    finish = False
    data = [-2,-2,-2]
    while(not finish):
        solvable = deterSolvability(prepoly)
        if solvable == True:
            for i in range(3):
                data.append(prepoly[i+3])
            # print(f"{prepoly[0:3]} \n")
            
        finish, curpoly = makePoly(prepoly)
        if(curpoly[2] != prepoly[2]):
            with open("solvable_poly.csv", "a", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(data)
                print(f"{data[0:3]}\n")
            data = list(curpoly[i] for i in range(3))

        prepoly = curpoly

def processData():
    maxColNum = 0
    data_file = "solvable_poly.csv"
    with open(data_file, 'r') as temp_f:
        # Read the lines
        lines = temp_f.readlines()

        for l in lines:
            # Count the column count for the current line
            column_count = len(l.split(','))
        
            # Set the new most column count
            maxColNum = column_count if maxColNum < column_count else maxColNum

    column_names = [i for i in range(0,maxColNum)]
    load_data = pd.read_csv(data_file, header=None, delimiter=",", names=column_names)
    # load_data.fillna(0, inplace=True)
    data = load_data.to_numpy()
    return data

def giveColor(Data):
    # 1. replace Nan to num cyclically
    row, col = Data.shape
    for r in range(row):
        count = 0
        cyclicComponent = []
        for i in Data[r,3:]:
            if not np.isnan(i):
                count+=1
                cyclicComponent.append(i)
        toFill = col - count-3
        nanStart = count+3
        haveFill = len(cyclicComponent)
        for index in range(nanStart, toFill+nanStart):
            Data[r,index] = cyclicComponent[index%haveFill]
    
    # 2. allocate corresponding color (-2,-1,0, 1,2 == 51,102,153,204,255)
    for r in range(row):
        for c in range(3, col):
            target = Data[r][c]
            if target == -2:    Data[r][c] = 51/255
            elif target == -1:    Data[r][c] = 102/255
            elif target == 0:    Data[r][c] = 153/255
            elif target == 1:    Data[r][c] = 204/255
            elif target == 2:    Data[r][c] = 255/255
    return Data
    
def randomSolving():
    n_solvable = 0
    while n_solvable < 100:
        poly = [random.randrange(-255, 256) for i in range(6)]
        if(deterSolvability(poly) is True):
            with open("solvable_polynomial.csv", "a", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(poly)
            n_solvable+=1
            print(n_solvable)

def postProcess():
    data_file = "solvable_polynomial.csv"
    load_data = pd.read_csv(data_file, header=None, delimiter=",")
    # load_data.fillna(0, inplace=True)
    data = load_data.to_numpy()
    return data

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])
    color = (data[3:6,num]+np.array([255,255,255]))/510
    line.set_color(color)

if __name__ =='__main__':
    # makeData()
    # Data= processData()
    # finalData = giveColor(Data)
    # drawPlot(finalData)
    # randomSolving()
    data = postProcess().T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    N = 100
    line, =ax.plot(data[0,0:1], data[1,0:1], data[2,0:1], 'o' )
    
    # Setting the axes properties
    ax.set_xlim3d([-255.0, 255.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-255.0, 255.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-255.0, 255.0])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, N, fargs=(data,line), interval= 100, blit = False)
    ani.save("ani.gif", writer="pillow")
    plt.show()

    


