import numpy as np

def rand_bbox(size, w_low=0.35, w_hight=0.45, h_low=0.25, h_high=0.35, se=0, time=0):
    H = size[0]
    W = size[1]

    cy = np.random.randint(H * h_low, H * h_high)
    cx = np.random.randint(W * w_low, W * w_hight)
    #�����������򲻳���������С

    bby1 = np.clip(cy , 0, H)
    bbx1 = np.clip(cx, 0, W)
    if se == 0:
        bbx1 += 10
        bby2 = np.clip(bby1 + bby1 % 15 + 10, 0, H)
        bbx2 = np.clip(bbx1 + bbx1 % 50 + 150, 0, W)
    elif se == 1:
        bby2 = np.clip(bby1 + bby1 % 5 + 8, 0, H)
        bbx2 = np.clip(bbx1 + bbx1 % 5 + 8, 0, W)
    elif se == 2:
        if time % 2 == 0:
            bby2 = np.clip(bby1 + bby1 % 50 + 150, 0, H)
            bbx2 = np.clip(bbx1 + bbx1 % 15 + 10, 0, W)     
        elif time % 2 == 1:
            bby2 = np.clip(bby1 + bby1 % 15 + 10, 0, H)
            bbx2 = np.clip(bbx1 + bbx1 % 50 + 150, 0, W)

    return bbx1, bby1, bbx2, bby2

def apply_glitch_effect_gs(image, intensity=1.0):
    glitched_image = image.copy()
    height, width, _ = image.shape

    for _ in range(int(intensity * 1000)):
        # blur
        [x1] = np.random.randint(1, width-1, 1)
        x2 = x1 + x1%10
        [y1] = np.random.randint(1, height-1, 1)
        y2 = y1 + y1%10
        color = image[y1, x1]
        glitched_image[y1:y2, x1:x2] = color

    return glitched_image

def cutout(image1, image2, image3):
    for i in range(10):
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(image2.shape, w_low=0, w_hight=1, h_low=0, h_high=1, se=2, time=1)
        image2[bby1_1:bby2_1, bbx1_1:bbx2_1, :] = image3[bby1_1:bby2_1, bbx1_1:bbx2_1, :].copy()
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(image2.shape, w_low=0, w_hight=1, h_low=0, h_high=1, se=2, time=1)
        image2[bby1_1:bby2_1, bbx1_1:bbx2_1, :] = image1[bby1_1:bby2_1, bbx1_1:bbx2_1, :].copy()  
    return image2

def fuzzy_cutout_fn(image1, image2, image3):
    # Apply the glitch effect
    image2 = cutout(image1, image2, image3)
    image2 = apply_glitch_effect_gs(image2, intensity=2)
    return image2

