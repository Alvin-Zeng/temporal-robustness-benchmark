import cv2
import numpy as np

def cover_process(image2, image1):
    x_scale = image2.shape[0] / image1.shape[0]
    y_scale = image2.shape[1] / image1.shape[1]
    scale = y_scale / 1.1 if x_scale > y_scale else x_scale / 1.1
    image1 = cv2.resize(image1,(int(image1.shape[1] * scale), int(image1.shape[0] * scale)))

    H, W, C = image2.shape
    H2, W2, C2 = image1.shape

    alpha_channel = image1[:, :, 3]
    alpha_channel = alpha_channel.astype(bool)
    alpha_channel = np.dstack([alpha_channel] * C)

    offset = []
    offset.append(H-H2)
    offset.append(0)

    image2[offset[0] : H2+offset[0], offset[1] : W2+offset[1], :C] = np.where(
                                                                        alpha_channel, 
                                                                        image1[:,:,:C],
                                                                        image2[offset[0] : H2+offset[0], offset[1] : W2+offset[1], :C])

    image2 = image2.astype(np.uint8)
    image2[:,:,:C] = cv2.medianBlur(image2[:,:,:C], 7)

    return image2

def cover(image2, image1):
    scale = 255 / (image2 + 1).max()
    image2 = scale * (image2 + 1) # (-1 ~ 1) --> (0 ~ 255)
    # cv2.imwrite('old.jpg', image2.copy())
    image2 = cover_process(image2, image1)
    # cv2.imwrite('new.jpg', image2)
    return image2 / scale - 1