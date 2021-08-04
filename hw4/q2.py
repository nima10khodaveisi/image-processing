import cv2
import dlib
from matplotlib import pyplot as plt
import math
import numpy as np


image1 = cv2.imread('resources/baba.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread('resources/sina.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# image3 = cv2.imread('resources/sina.jpg')
# image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
# image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))

def draw_triangles(img, trianglesList):
    image = img.copy()
    for p in trianglesList:
        p1 = (p[0], p[1])
        p2 = (p[2], p[3])
        p3 = (p[4], p[5])
        cv2.line(image, p1, p2, (255, 0, 0), 1)
        cv2.line(image, p2, p3, (255, 0, 0), 1)
        cv2.line(image, p3, p1, (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def get_key_points_of_face(image):
    global detector, predictor
    faces = detector(image)
    points = []
    for face in faces:
        for p in range(68):
            landmarks = predictor(image, box=face)
            x = landmarks.part(p).x
            y = landmarks.part(p).y
            points.append((x, y))
    points.append((0, 0))
    points.append((0, image.shape[0] - 1))
    points.append((image.shape[1] - 1, 0))
    points.append((image.shape[1] - 1, image.shape[0] - 1))
    points.append((math.floor(image.shape[1] / 2), 0))
    points.append((math.floor(image.shape[1] / 2), image.shape[0] - 1))
    points.append((0, math.floor(image.shape[0] / 2)))
    points.append((image.shape[1] - 1, math.floor(image.shape[1] / 2)))
    return points


def get_triangles_list(image, points):
    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    indexes = []
    trianglesList = subdiv.getTriangleList()
    for p in trianglesList:
        p1 = (p[0], p[1])
        p2 = (p[2], p[3])
        p3 = (p[4], p[5])
        ind1 = points.index(p1)
        ind2 = points.index(p2)
        ind3 = points.index(p3)
        indexes.append((ind1, ind2, ind3))
    return trianglesList, indexes


def get_triangular_images(image1, image2):
    points1 = get_key_points_of_face(image1)
    points2 = get_key_points_of_face(image2)
    trianglesList1, inds = get_triangles_list(image1, points1)
    trianglesList2 = []
    for ind in inds:
        ind1 = ind[0]
        ind2 = ind[1]
        ind3 = ind[2]
        trianglesList2.append((points2[ind1][0], points2[ind1][1], points2[ind2][0]
                               , points2[ind2][1], points2[ind3][0], points2[ind3][1]))

    return trianglesList1, trianglesList2, inds, points1, points2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('res02.mp4',fourcc, 15, (image1.shape[1], image1.shape[0]))

def morphImages(image1, image2, result_image, trianglesList1, trianglesList2, trianglesList3, t):
    for i in range(trianglesList3.__len__()):
        p = trianglesList1[i]
        p1 = (p[0], p[1])
        p2 = (p[2], p[3])
        p3 = (p[4], p[5])
        tri1 = [p1, p2, p3]

        p = trianglesList2[i]
        p1 = (p[0], p[1])
        p2 = (p[2], p[3])
        p3 = (p[4], p[5])
        tri2 = [p1, p2, p3]

        p = trianglesList3[i]
        p1 = (p[0], p[1])
        p2 = (p[2], p[3])
        p3 = (p[4], p[5])
        tri3 = [p1, p2, p3]

        mask = np.zeros(result_image.shape, dtype='uint8')
        cv2.fillConvexPoly(mask, np.int32(tri3), (1, 1, 1))
        # plt.imshow(mask)
        # plt.show()
        mat1 = cv2.getAffineTransform(np.float32(tri1), np.float32(tri3))
        affineImage1 = cv2.warpAffine(image1, mat1, (image1.shape[1], image1.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mat2 = cv2.getAffineTransform(np.float32(tri2), np.float32(tri3))
        affineImage2 = cv2.warpAffine(image2, mat2, (image2.shape[1], image2.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # plt.imshow(affineImage2)
        # plt.show()
        result_image[mask > 0] = (1 - t) * affineImage1[mask > 0] + t * affineImage2[mask > 0]
    video.write(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


def face_morphing(image1, image2):
    T = 0.01
    trianglesList1, trianglesList2, inds, points1, points2 = get_triangular_images(image1, image2)
    # draw_triangles(image1, trianglesList1)
    # draw_triangles(image2, trianglesList2)
    result_image = np.zeros(image1.shape, dtype='uint8')
    t = 0
    while t <= 1:
        points = []
        for i in range(points1.__len__()):
            x = (1 - t) * points1[i][0] + t * points2[i][0]
            y = (1 - t) * points1[i][1] + t * points2[i][1]
            points.append((math.floor(x), math.floor(y)))
        trianglesList3 = []
        for ind in inds:
            ind1 = ind[0]
            ind2 = ind[1]
            ind3 = ind[2]
            trianglesList3.append((points[ind1][0], points[ind1][1], points[ind2][0]
                                   , points[ind2][1], points[ind3][0], points[ind3][1]))
        morphImages(image1, image2, result_image, trianglesList1, trianglesList2, trianglesList3, t)
        t += T


face_morphing(image1, image2)
# face_morphing(image2, image3)
cv2.waitKey(0)
cap.release()
video.release()
cv2.destroyAllWindows()