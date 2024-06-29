import geojson
from shapely.geometry import shape, Point
import random
import numpy as np
from PIL import Image
from scipy.interpolate import griddata

ntasfile = open("ntas.geojson")

ntas = geojson.load(ntasfile)

minlong = ntas['features'][0]['geometry']['coordinates'][0][0][0][0]
maxlong = ntas['features'][0]['geometry']['coordinates'][0][0][0][0]
minlat = ntas['features'][0]['geometry']['coordinates'][0][0][0][1]
maxlat = ntas['features'][0]['geometry']['coordinates'][0][0][0][1]

for feature in ntas['features']:
    for polygon in feature['geometry']['coordinates']:
        for linearring in polygon:
            for coordinate in linearring:
                minlong = min(minlong, coordinate[0])
                maxlong = max(maxlong, coordinate[0])
                minlat = min(minlat, coordinate[1])
                maxlat = max(maxlat, coordinate[1])
                
print("minlong: "+ str(minlong))
print("maxlong: " + str(maxlong))
print("minlat: " + str(minlat))
print("maxlat: " + str(maxlat))

difflong = maxlong - minlong
difflat = maxlat - minlat
print("difflong: " + str(difflong))
print("difflat: " + str(difflat))
print(len(ntas['features']))

# 2345 pixels by 2334 pixels
# temps.tif is 2061 by 1556 pixels

tempsfile = Image.open("temps.tif")
tempsarr = np.array(tempsfile)

# Define the new dimensions
new_width = 2345
new_height = 2334

# Create a grid of coordinates for the new dimensions
x = np.linspace(0, 2060, new_width)
print(x)
y = np.linspace(0, 1555, new_height)
print(y)
# Interpolate the values from tempsarr to the new dimensions
oldpoints = np.array(np.meshgrid(np.arange(1556), np.arange(2061))).T.reshape(-1, 2)
newpoints = np.array(np.meshgrid(y, x)).T.reshape(-1, 2)
temps = griddata(oldpoints, tempsarr.reshape(-1), newpoints, method="linear")
print(temps)

# with open("")