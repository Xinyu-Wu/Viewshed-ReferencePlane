# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:32:10 2017

@author: wuxinyu
"""

import numpy as np
import arcpy
import threading
import copy
import time

class ReferencePlaneLocalArea():
    # help matrixs: lowest elevation matrix & visibility matrix
    def __init__(self, rows, columns, elevation):
        self.rows = rows
        self.columns = columns
        self.elevation = copy.deepcopy(elevation)

    def run(self, view_row, view_column):
        #split
        # print 'spliting:', time.time()
        left_up = self.elevation[:view_row+1, :view_column+1]
        left_bottom = self.elevation[view_row:, :view_column+1]
        right_up = self.elevation[:view_row+1, view_column:]
        right_bottom = self.elevation[view_row:, view_column:]
                         
        #reverse
        # print 'reversing:', time.time()
        right_up = np.flipud(right_up)
        left_bottom = np.fliplr(left_bottom)
        left_up = np.fliplr(left_up)
        left_up = np.flipud(left_up)
        threads = []
        threads.append(threading.Thread(target=referenceline,
                                        args=(self, right_up, 'right_up')))
        threads.append(threading.Thread(target=referenceline,
                                        args=(self, left_bottom, 'left_bottom')))
        threads.append(threading.Thread(target=referenceline,
                                        args=(self, left_up, 'left_up')))
        threads.append(threading.Thread(target=referenceline,
                                        args=(self, right_bottom, 'right_bottom')))
        # print 'dividing:',time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # print 'finish computation:',time.time()
        right_up_v = np.flipud(self.right_up_v)
        left_bottom_v = np.fliplr(self.left_bottom_v)
        left_up_v = np.flipud(self.left_up_v)
        left_up_v = np.fliplr(left_up_v)
        
        #combine results together
        up = np.hstack((left_up_v, right_up_v[:,1:]))
        bottom = np.hstack((left_bottom_v, self.right_bottom_v[:,1:]))
        self.result = copy.deepcopy(np.vstack((up, bottom[1:,:])))
        
    def getVisibility(self):
        return np.array(self.result)
        
def referenceline(self, localE, sector):
    # print 'threadx started:', time.time()
    width = len(localE[0])
    height = len(localE)
    lowestE = [[0 for i in range(width)] for j in range(height)]
    lowestE[0][0] = localE[0][0]
    lowestE[1][0] = localE[1][0]
    lowestE[0][1] = localE[0][1]
    lowestE[1][1] = localE[1][1]  # treat default height as lowest visible elevation
    view_elevation = localE[0][0]

    visible = [[0 for i in range(width)] for j in range(height)]
    visible[0][0] = 1
    visible[1][0] = 1
    visible[0][1] = 1
    visible[1][1] = 1 # adjacency grid is visible
    
    # calculate visibility with X-axis
    elevation_x = [localE[0][i] for i in range(width)]
    for i in range(2, width):
        z = view_elevation + float(i) / (i-1) * (lowestE[0][i-1] - view_elevation) # lowest visible elevation
        if elevation_x[i] > z:
            visible[0][i] = 1
            lowestE[0][i] = elevation_x[i]
        else:
            lowestE[0][i] = z

    # calculate visibility with Y-axis
    elevation_y = [localE[i][0] for i in range(height)]
    for i in range(2, height):
        z = view_elevation + float(i) / (i-1) * (lowestE[i-1][0] - view_elevation) # lowest visible elevation
        if elevation_y[i] > z:
            visible[i][0] = 1
            lowestE[i][0] = elevation_y[i]
        else:
            lowestE[i][0] = z

    # calculate visibilty with diagonal
    height_min = False # width < height ?
    min_diag = 0
    if(width >= height):
        min_diag = height
        height_min = True
    else:
        min_diag = width
    elevation_diag = [localE[i][i] for i in range(min_diag)]
    for i in range(2, min_diag):
        z = view_elevation + float(i) / (i-1) * (lowestE[i-1][i-1] - view_elevation)
        if elevation_diag[i] > z:
            visible[i][i] = 1
            lowestE[i][i] = elevation_diag[i]
        else:
            lowestE[i][i] = z
            
    
    trs_E = np.transpose(localE)
    trs_lowestE = np.transpose(lowestE)
    trs_visible = np.transpose(visible)
    threads = []
    threads.append(threading.Thread(target=clc_visibilityInSector,
                                    args=(localE, lowestE, visible, view_elevation, min_diag, height_min)))
    threads.append(threading.Thread(target=clc_visibilityInSector,
                                    args=(trs_E, trs_lowestE, trs_visible, view_elevation, min_diag, ~height_min)))
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    # transpose back 
    trs_visible = np.transpose(trs_visible)
    if sector == 'left_up':
        self.left_up_v = np.logical_or(np.mat(visible), np.mat(trs_visible))*1
    elif sector == 'left_bottom':
        self.left_bottom_v = np.logical_or(np.mat(visible), np.mat(trs_visible))*1
    elif sector == 'right_up':
        self.right_up_v = np.logical_or(np.mat(visible), np.mat(trs_visible))*1
    elif sector == 'right_bottom':
        self.right_bottom_v = np.logical_or(np.mat(visible), np.mat(trs_visible))*1
    # print 'threadx ended:', time.time()

def clc_visibilityInSector(localE, lowestE, visible, view_elevation, min_diag, height_min):
    width = len(localE[0])
    height = len(localE)
    # (m-1,n-1) (m-1,n)

    if height_min:
        #lower triangular matrix    
        for i in range(2, min_diag):
            for j in range(1, i+1):
                z31 = lowestE[i-1][j-1] - view_elevation
                z21 = lowestE[i-1][j] - view_elevation
                x31 = i-1
                x21 = i-1
                y31 = j-1
                y21 = j
                z = float(view_elevation) - (float(i)*(float(y21*z31) - float(y31*z21)) +
                                      float(j)*(float(z21*x31) - float(z31*x21))) / \
                                      float(x21*y31-x31*y21)
                if localE[i][j] > z:
                    visible[i][j] = 1
                    lowestE[i][j] = localE[i][j]
                else:
                    lowestE[i][j] = z
    else:
        for i in range(2, height):
            if i <= min_diag:
                for j in range(1, i+1):
                    z31 = lowestE[i-1][j-1] -view_elevation
                    z21 = lowestE[i-1][j] - view_elevation
                    x31 = i-1
                    x21 = i-1
                    y31 = j-1
                    y21 = j
                    z = float(view_elevation) - (float(i)*(float(y21*z31) - float(y31*z21)) +
                                          float(j)*(float(z21*x31) - float(z31*x21))) / \
                                          float(x21*y31-x31*y21)
                    if localE[i][j] > z:
                        visible[i][j] = 1
                        lowestE[i][j] = localE[i][j]
                    else:
                        lowestE[i][j] = z
            else:
                for j in range(1, width):
                    z31 = lowestE[i-1][j-1] -view_elevation
                    z21 = lowestE[i-1][j] - view_elevation
                    x31 = i-1
                    x21 = i-1
                    y31 = j-1
                    y21 = j
                    z = float(view_elevation) - (float(i)*(float(y21*z31) - float(y31*z21)) +
                                          float(j)*(float(z21*x31) - float(z31*x21))) / \
                                          float(x21*y31-x31*y21)
                if localE[i][j] > z:
                    visible[i][j] = 1
                    lowestE[i][j] = localE[i][j]
                else:
                    lowestE[i][j] = z
                
                
if __name__ == '__main__':
    # import file and getAttributes
    path = arcpy.GetParameterAsText(0)
    point_text = arcpy.GetParameterAsText(1)
    defaultheight = float(arcpy.GetParameterAsText(2))
    output = arcpy.GetParameterAsText(3)
    raster = arcpy.Raster(path)
    lon = float(point_text.split(' ')[0])
    lat = float(point_text.split(' ')[1])
    viewpoint = arcpy.Point(lon, lat)
    # defaultheight=0.0
    describe = arcpy.Describe(raster)
    cell_size_x = describe.meanCellWidth
    cell_size_y = describe.meanCellHeight
    spatialReference = describe.spatialReference
    extent = describe.Extent
    rows = raster.height
    columns = raster.width
    column = int(float(viewpoint.X-extent.XMin)/(extent.XMax-extent.XMin)*columns)
    row = int(float(viewpoint.Y-extent.YMax)/(extent.YMin-extent.YMax)*rows)
    lower_left = arcpy.Point(X=extent.XMin, Y=extent.YMin)
    
    #initialize
    arrays = arcpy.RasterToNumPyArray(raster)
    arrays[row][column] += defaultheight
    rp = ReferencePlaneLocalArea(rows, columns, arrays)
    rp.run(row, column)
    result = rp.getVisibility()
    result = arcpy.NumPyArrayToRaster(result, lower_left_corner=lower_left, x_cell_size=cell_size_x, y_cell_size=cell_size_y)
    arcpy.DefineProjection_management(result, spatialReference)
    # print 'saving:', time.time()
    result.save(output)
    # print 'project done:', time.time()
