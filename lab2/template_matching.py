import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.metrics import structural_similarity as ssim

IMG_NAME = "img/pic4_camera.jpg"
SUB_IMG_NAME = "img/pic4_frag.jpg"

def find_window(full_img, sub_img, threshold=0.5):
    full_h, full_w = full_img.shape[:2]
    sub_h, sub_w = sub_img.shape[:2]

    for y in range(full_h - sub_h + 1):
        for x in range(full_w - sub_w + 1):
            window = full_img[y:y + sub_h, x:x + sub_w]
            score = ssim(sub_img, window)

            if score > threshold:
                return (x, y)
            
    print('sub img not found')
    return None

full_img = cv2.imread(IMG_NAME, cv2.IMREAD_GRAYSCALE)
sub_img = cv2.imread(SUB_IMG_NAME, cv2.IMREAD_GRAYSCALE)

win_pos = find_window(full_img, sub_img)

if win_pos:
    print(f'window pos {win_pos}')
    sub_h, sub_w = sub_img.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(img.imread(IMG_NAME))
    rect = patches.Rectangle(win_pos, sub_w, sub_h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.savefig('pic4_tm_camera_results_.png')
    plt.show()
else:
    print('sub not found at all')