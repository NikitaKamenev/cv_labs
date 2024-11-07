import cv2 as cv
import time
import matplotlib.pyplot as plt
import os

output_dir = 'binarized_images_lib'
os.makedirs(output_dir, exist_ok=True)

image_paths = {
    "sd": 'images/sd.jpg',
    "hd": 'images/hd.jpg',
    "full_hd": 'images/full_hd.jpg',
    "2k": 'images/2k.jpg',
    "4k": 'images/4k.jpg'
}

execution_times = {}

block_size = 25
C = 7

for resolution, path in image_paths.items():
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image at path '{path}' could not be found or loaded.")
        continue

    start = time.time()
    th_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C)
    end = time.time()

    elapsed_time = end - start
    execution_times[resolution] = elapsed_time

    output_path = os.path.join(output_dir, f'adaptive_binarized_{resolution}.jpg')
    cv.imwrite(output_path, th_img)

plt.figure(figsize=(10, 6))
plt.plot(execution_times.keys(), execution_times.values(), marker='o')
plt.xlabel('Resolution')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Adaptive Thresholding Time vs Image Resolution')
plt.grid(True)

plot_path = os.path.join(output_dir, 'lib_binarization_time_plot.png')
plt.savefig(plot_path)

plt.show()
