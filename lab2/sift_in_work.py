import numpy as np
import cv2
import sift as SIFT
from matplotlib import pyplot as plt
import logging

# Константы
BASE_IMG = 'img/pic4_camera.jpg'
SUB_IMG_NAME = 'img/pic4_frag.jpg'
MIN_MATCH_COUNT = 10

# Логгер
logger = logging.getLogger(__name__)

# Загрузка изображений
img1 = cv2.imread(SUB_IMG_NAME, 0)  # queryImage
img2 = cv2.imread(BASE_IMG, 0)  # trainImage

# Вычисление ключевых точек и дескрипторов SIFT
kp1, des1 = SIFT.computeKeypointsAndDescriptors(img1)
kp2, des2 = SIFT.computeKeypointsAndDescriptors(img2)

# Инициализация и использование FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Тест Лоу
good = [m for m, n in matches if m.distance < 0.7 * n.distance]

if len(good) > MIN_MATCH_COUNT:
    # Оценка гомографии между шаблоном и сценой
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Рисование обнаруженного шаблона на сцене
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Создание итогового изображения
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h2 - h1) // 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Отображение и сохранение итогового изображения
    plt.imshow(newimg)
    plt.axis('off')  # Отключение осей
    plt.show()
    cv2.imwrite('pic4_sift_camera_result.jpg', newimg)
else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
