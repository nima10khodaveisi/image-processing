import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import math

file = open('resources/Points.txt')
n = int(file.readline())
points = []
for i in range(n):
    line = file.readline()
    arr = line.split(' ')
    x = float(arr[0])
    y = float(arr[1])
    points.append((x , y))


for i in range(n):
    plt.plot(points[i][0], points[i][1], 'ro')
plt.savefig('res01.jpg')
plt.show()
plt.clf()


kmean = KMeans(n_clusters=2).fit(points)
lables = kmean.labels_
for i in range(n):
    label = lables[i]
    color = (1, 0, 0)
    if label == 1:
        plt.plot(points[i][0], points[i][1], 'ro')
    else:
        plt.plot(points[i][0], points[i][1], 'bo')
plt.savefig('res02.jpg')
plt.show()

ghotbi_points = []

for i in range(n):
    x = points[i][0]
    y = points[i][1]
    nx = math.sqrt(x ** 2 + y ** 2)
    ny = math.atan(y / x)
    ghotbi_points.append((nx , ny))

plt.clf()
ghotbi = KMeans(n_clusters=2).fit(ghotbi_points)
lables = ghotbi.labels_
for i in range(n):
    label = lables[i]
    if label == 1:
        plt.plot(points[i][0], points[i][1], 'ro')
    else:
        plt.plot(points[i][0], points[i][1], 'bo')
plt.savefig('res03.jpg')

plt.show()


