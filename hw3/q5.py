import cv2
from matplotlib import pyplot as plt
import math
import numpy as np

isClicking = False
hasBeenStarted = False
points = []
org_image = cv2.imread('tasbih.jpg')
image = org_image.copy()
init_points_image = image.copy()


def mouse_click(event, x, y, flags, param):
    global isClicking, hasBeenStarted, points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        isClicking = True
        hasBeenStarted = True
    elif hasBeenStarted and event == cv2.EVENT_LBUTTONUP:
        isClicking = False
    elif isClicking and event == cv2.EVENT_MOUSEMOVE:
        points.append((y, x))
        cv2.circle(init_points_image, (x, y), 1, (0, 0, 1), -1)


def dist(x0, y0, x1, y1, dr):
    val = (x0 - x1) * (x0 - x1)
    val += (y0 - y1) * (y0 - y1)
    val = math.sqrt(val)
    return (val - dr) * (val - dr)

object = cv2.VideoWriter('contour.mp4', -1, 30, (image.shape[1], image.shape[0]))

def show(iteration):
    global image, points
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    draw_points = []
    for c in points:
        x = c[0]
        y = c[1]
        draw_points.append((y, x))
    cv2.polylines(img, np.int32([draw_points]), True, color=(0, 0, 0))
    object.write(img)
    if iteration == 1:
        plt.imshow(img)
        plt.show()
        plt.imsave('res09.jpg', img)


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_click)

# end when hasBeenStarted = True and isClicking = False

while not hasBeenStarted or isClicking:
    cv2.imshow('image', init_points_image)
    cv2.waitKey(1)

cv2.destroyAllWindows()

cp_points = points.copy()
points = []
for i in range(0, cp_points.__len__(), 3):
    points.append(cp_points[i])

sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
gradient = sobelx ** 2 + sobely ** 2
gradient[gradient < 5e6] = 0


delta = 20
alpha = 1
center_zarib = 0.5

iteration = 180

n = points.__len__()

while iteration > 0:
    show(iteration)
    iteration = iteration - 1
    center = [0, 0]
    d = 0
    for i in range(n):
        c0 = points[i]
        center[0] += c0[0]
        center[1] += c0[1]
        c1 = points[(i+1) % n]
        d += math.sqrt((c0[0] - c1[0]) * (c0[0] - c1[0]) + (c0[1] - c1[1]) * (c0[1] - c1[1]))
    d /= n
    d *= 0.8
    center[0] /= n
    center[1] /= n
    dp = np.zeros((n, 9), dtype='float32')
    par = np.zeros((n, 9), dtype='int8')
    par.fill(-1)
    dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dy = [-1, 1, 0, -1, 1, 0, -1, 1, 0]
    best_points = []
    par0 = np.zeros(9, dtype='int8')
    par0.fill(-1)
    dp0 = np.zeros(9, dtype='float32')
    for i in range(9):
        dp0[i] = 1e10
    for dif0 in range(9): #fix the first one
        for i in range(1, n):
            x = points[i][0]
            y = points[i][1]
            for dif in range(9):
                nx = x + dx[dif]
                ny = y + dy[dif]
                dp[i, dif] = 1e10
                for bef in range(9):
                    if i == 1:
                        bef = dif0
                    bx = points[i - 1][0]
                    by = points[i - 1][1]
                    bnx = bx + dx[bef]
                    bny = by + dy[bef]
                    sum_gradient = gradient[nx, ny, 0] + gradient[nx, ny, 1] + gradient[nx, ny, 2]
                    value = 0
                    if sum_gradient != 0:
                        value = -(delta / math.log(sum_gradient)) * sum_gradient
                    value += alpha * dist(nx, ny, bnx, bny, d)
                    value += center_zarib * dist(nx, ny, center[0], center[1], 0)
                    if dp[i, dif] > value + dp[i - 1, bef]:
                        dp[i, dif] = value + dp[i - 1, bef]
                        par[i, dif] = bef
                if i == n - 1:
                    bx = points[0][0]
                    by = points[0][1]
                    bnx = bx + dx[dif0]
                    bny = by + dy[dif0]
                    value = alpha * dist(nx, ny, bnx, bny, d)
                    sum_gradient = gradient[bnx, bny, 0] + gradient[bnx, bny, 1] + gradient[bnx, bny, 2]
                    if sum_gradient != 0:
                        value += -(delta / math.log(sum_gradient)) * sum_gradient
                    value += dist(bnx, bny, center[0], center[1], 0) * center_zarib
                    if dp0[dif0] > dp[i, dif] + value:
                        dp0[dif0] = dp[i, dif] + value
                        par0[dif0] = dif
    ind = 0
    for dif in range(9):
        if dp0[dif] < dp0[ind]:
            ind = dif
    best_points.append((points[0][0] + dx[ind], points[0][1] + dy[ind]))
    ind = par0[ind]
    cur = n - 1
    other_points = []
    while cur >= 1:
        other_points.append((points[cur][0] + dx[ind], points[cur][1] + dy[ind]))
        ind = par[cur, ind]
        cur = cur - 1
    for i in range(other_points.__len__() - 1, -1, -1):
        best_points.append(other_points[i])
    points = []
    for c in best_points:
        points.append(c)

object.release()
cv2.destroyAllWindows()
