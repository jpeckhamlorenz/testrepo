#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:19:57 2025

@author: leoallentoff
"""

import numpy as np
import matplotlib.pyplot as plt

def move2gcode(xstart, xstop, ystart, ystop, zstart, zstop, linespacing, nozzlesize, layerheight):
    xdistance = xstop - xstart
    ydistance = ystop - ystart
    zdistance = zstop - zstart
    
    totdistance = np.sqrt(xdistance**2 + ydistance**2 + zdistance**2)
    linenum = totdistance/linespacing
    
    xmove = np.linspace(xstart, xstop, int(np.ceil(linenum))+1)
    ymove = np.linspace(ystart, ystop, int(np.ceil(linenum))+1)
    zmove = np.linspace(zstart, zstop, int(np.ceil(linenum))+1)
    
    distancetraveled = np.sqrt((xmove-xstart)**2 + (ymove-ystart)**2 + (zmove-zstart)**2)
    
    if len(set(layerheight)) == 1:
        layerheight = np.zeros(len(distancetraveled)) + layerheight[0]
    
    InstantVolume = nozzlesize * layerheight * distancetraveled[1]
    
    Volume = np.cumsum(InstantVolume)
    
    Volume = np.insert(Volume[0:-1], 0, 0)
    
    return(xmove, ymove, zmove, Volume)

def combinemoves(xstart, xstop, ystart, ystop, zstart, zstop, linespacing, nozzlesize, layerheight):
    X = np.array([])
    Y = np.array([])
    Z = np.array([])
    E = np.array([])
    
    for i, x in enumerate(xstart):
        [Xi, Yi, Zi, Ei] = move2gcode(xstart[i], xstop[i], ystart[i], ystop[i], zstart[i], zstop[i], linespacing[i], nozzlesize[i], layerheight[i])
        
        if i == 0:
            Efix = Ei
            
            X = np.concatenate([X, Xi])
            Y = np.concatenate([Y, Yi])
            Z = np.concatenate([Z, Zi])
            E = np.concatenate([E, Efix])
            
        else:
            Efix = Ei + E[-1]
            
            X = np.concatenate([X, Xi[1:]])
            Y = np.concatenate([Y, Yi[1:]])
            Z = np.concatenate([Z, Zi[1:]])
            E = np.concatenate([E, Efix[1:]])
    
    return(X, Y, Z, E*5)

def linetest(linenumber, speed, times, flows, linespacing):
   
    xmax = speed/60 * np.amax(times) + 10
   
    xstart = np.array([xmax, xmax, 10, 10] * linenumber)
    xstop = np.array([10,10, xmax, xmax] * linenumber)
    
    xstart = xstart[1:(linenumber*2)]
    xstop = xstop[0:(linenumber*2-1)]
    
    yspacing = linespacing
    ytemp = np.zeros((linenumber*2))
                     
    ytemp = np.arange(240 - (yspacing*(linenumber-1)), 241, yspacing)
    ytemp = np.repeat(ytemp[::-1], 2)
        
    ystart = ytemp[0:-1]
    ystop = ytemp[1:]
    
    
    zstart = np.zeros(len(xstart)) + 0.3
    zstop = np.zeros(len(xstop)) + 0.3
   
    linespacing = np.zeros(len(xstart)) + speed/60*0.05
    
    nozzlesize = np.zeros(len(xstart)) + 0.25
    
    desiredtimes = np.linspace(0, np.amax(times), int(np.ceil((xmax-10)/linespacing[0]))+1)
    
    interpflows = np.zeros((len(flows[:, 0]), len(desiredtimes)))
    
    for i, row in enumerate(flows):
    
        interpflows[i] = np.interp(desiredtimes, times[i], flows[i])
        
    ymoves = np.arange(1, linenumber)
        
    layerheight = np.insert(interpflows, ymoves, 0, axis=0)
    
    [X, Y, Z, E] = combinemoves(xstart, xstop, ystart, ystop, zstart, zstop, linespacing, nozzlesize, layerheight)
    
    return(X, Y, Z, E)


#Manually creating lines with constant flows

#xstart = np.array([280, 10, 10, 280, 280, 10, 10, 280, 280, 10, 10])
#xstop = np.array([10, 10, 280, 280, 10, 10, 280, 280, 10, 10, 280])
#ystart = np.array([240, 240, 192, 192, 144, 144, 96, 96, 48, 48, 0])
#ystop = np.array([240, 192, 192, 144, 144, 96, 96, 48, 48, 0, 0])
#zstart = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
#zstop = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
#linespacing = np.array([0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625])
#nozzlesize = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
#layerheight = np.array([0.1, 0, 0.3, 0, 0.5, 0, 0.2, 0, 0.05, 0, 0.4])

#[X, Y, Z, E] = combinemoves(xstart, xstop, ystart, ystop, zstart, zstop, linespacing, nozzlesize, layerheight)


#Automatically creating lines with inputted time and variable flow data

timedata1 = np.linspace(0, 20, 100) #Put time data for each line here
timedata2 = np.linspace(0, 20, 100)
timedata3 = np.linspace(0, 20, 100)
timedata4 = np.linspace(0, 20, 100)
timedata5 = np.linspace(0, 20, 100)

flowdata1 = np.linspace(2, 1.5, 100) #Put flow data for each line here
flowdata2 = np.linspace(1.5, 1, 100)
flowdata3 = np.linspace(1, 0.5, 100)
flowdata4 = np.linspace(0.5, 0, 100)
flowdata5 = np.concatenate([np.linspace(2, 0, 50), np.linspace(2, 0, 50)])

timedata = np.array([timedata1, timedata2, timedata3, timedata4, timedata5])
flowdata = np.array([flowdata1, flowdata2, flowdata3, flowdata4, flowdata5])

speed = 750

linenumber = len(flowdata[:, 0])

linespacing = 40

[X, Y, Z, E] = linetest(linenumber, speed, timedata, flowdata, linespacing)

plt.close('all')
plt.figure(1)
plt.plot(X, Y)

GCodeNumbers = np.column_stack([X, Y, Z, E])

GCodeNumbers = np.round(GCodeNumbers, decimals = 4)

GCode = np.array(np.vectorize(str)(GCodeNumbers), dtype="object")

GCode[:, 0] = 'G1 X' + GCode[:, 0]
GCode[:, 1] = 'Y' + GCode[:, 1]
GCode[:, 2] = 'Z' + GCode[:,2]
GCode[:, 3] = 'E' + GCode[:, 3] + ' F' + np.vectorize(str)(speed) +';'

with open("GCode Setup Start.txt", "r") as inputStart:
    with open("GCode Setup End.txt", "r") as inputEnd:

        with open("LineTestThinner.gcode", "w") as output: #Change GCode file name
    
            for line in inputStart: 
                output.write(line)   
    
            for line in GCode:
                output.write(" ".join(line) + "\n")
                
            for line in inputEnd: 
                output.write(line) 
                
                
                