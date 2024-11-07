import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import os

output_dir = 'binarized_images'
os.makedirs(output_dir, exist_ok=True)

image_paths = {
    "sd": 'images/sd.jpg',
    "hd": 'images/hd.jpg',
    "full_hd": 'images/full_hd.jpg',
    "2k": 'images/2k.jpg',
    "4k": 'images/4k.jpg'
}

KERNEL_SIZE = (25, 25)
C = 7
h_k = KERNEL_SIZE[0] // 2

execution_times = {}

for resolution, path in image_paths.items():
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image at path '{path}' could not be found or loaded.")
        continue

    M, N = img.shape

    pdimg = np.pad(img, h_k, 'minimum')
    cp_img = pdimg.copy()

    start = time.time()
    for i in range(h_k - 1, M + h_k):
        for j in range(h_k - 1, N + h_k):
            m = pdimg[i - h_k + 1:i + h_k, j - h_k + 1:j + h_k].mean() - C
            cp_img[i, j] = 0 if pdimg[i, j] < m else 255
    end = time.time()

    elapsed_time = end - start
    execution_times[resolution] = elapsed_time

    output_path = os.path.join(output_dir, f'binarized_{resolution}.jpg')
    cv.imwrite(output_path, cp_img[h_k:-h_k, h_k:-h_k])

plt.figure(figsize=(10, 6))
plt.plot(execution_times.keys(), execution_times.values(), marker='o')
plt.xlabel('Resolution')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Binarization Time vs Image Resolution')
plt.grid(True)

plot_path = os.path.join(output_dir, 'binarization_time_plot.png')
plt.savefig(plot_path)

plt.show()

