import json
from PIL import Image
import cv2
import os
import numpy as np
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# class MotionImage(WandImage):
#     def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
#         wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

# def motion_blur(x, severity=1):
#     c = [(5, 2), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

#     _, image_buffer = cv2.imencode('.png', x)
#     x = MotionImage(blob=image_buffer.tobytes())
#     x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
#     x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#     return np.clip(x, 0, 255)

# def cover_process(image2, image1):
#     x_scale = image2.shape[0] / image1.shape[0]
#     y_scale = image2.shape[1] / image1.shape[1]
#     scale = y_scale / 1.1 if x_scale > y_scale else x_scale / 1.1
#     image1 = cv2.resize(image1,(int(image1.shape[1] * scale), int(image1.shape[0] * scale)))

#     H, W, C = image2.shape
#     H2, W2, C2 = image1.shape

#     alpha_channel = image1[:, :, 3]
#     alpha_channel = alpha_channel.astype(bool)
#     alpha_channel = np.dstack([alpha_channel] * C)

#     offset = []
#     offset.append(H-H2)
#     offset.append(0)

#     image2[offset[0] : H2+offset[0], offset[1] : W2+offset[1], :C] = np.where(
#                                                                         alpha_channel, 
#                                                                         image1[:,:,:C],
#                                                                         image2[offset[0] : H2+offset[0], offset[1] : W2+offset[1], :C])

#     image2 = image2.astype(np.uint8)
#     image2[:,:,:C] = cv2.medianBlur(image2[:,:,:C], 7)

#     return image2

# def cover(image2, image1):
#     scale = 255 / (image2 + 1).max()
#     image2 = scale * (image2 + 1) # (-1 ~ 1) --> (0 ~ 255)
#     # cv2.imwrite('old.jpg', image2.copy())
#     image2 = cover_process(image2, image1)
#     cv2.imwrite("/home/xiaoyong/TriDet/p8.jpg", image2)
#     return image2 / scale - 1

# cover_img = cv2.imread(os.path.join("/home/xiaoyong/corrupt-Thumos14/libs/cover_img/hand.png"), -1)
# origin_img = cv2.imread(os.path.join("/home/xiaoyong/TriDet/p6.jpg"), -1)
# cover(origin_img, cover_img)

def read_frame_from_mp4(mp4_file_path, frame_number):

    save_image = True
    output_path = '/home/xiaoyong/TriDet/p9.jpg'
    video = cv2.VideoCapture(mp4_file_path)

    if not video.isOpened():
        print("unable to open video")
        return


    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


    ret, frame = video.read()
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    

    frame_pil.save(output_path)
    video.release()


# mp4_file_path = "/root/autodl-tmp/thumos/video/test_213_videos/video_test_0000045.mp4"
# frame_number = 3010 


# read_frame_from_mp4(mp4_file_path, frame_number)

data = np.load('/root/autodl-tmp/thumos_feature/i3d/light_180/action_length_5_percentage/thumos/i3d_features/video_test_0001098.npy')
print(data.shape)
# clean_list = []
# mask_list = []
# clean_score = []
# mask_score = []
    
# with open('/home/xiaoyong/TriDet/annotations/thumos14.json', 'r') as anno_f:
#     anno_data = json.load(anno_f)

# with open('/home/xiaoyong/TriDet/ckpt_ori/thumos_videomae_clean/mask/action_length_1_percentage/eval_results.json', 'r') as clean_f:
#     clean_pred = json.load(clean_f)

# with open('/home/xiaoyong/TriDet/ckpt_jsd/w_1/thumos_videomae_random1/mask/action_length_1_percentage/eval_results.json', 'r') as mask_f:
#     mask_pred = json.load(mask_f)

# print(anno_data['database']['video_test_0000045']['annotations'][0], '\n')

# # print(clean_pred['results']['video_test_0000045'][1], '\n')
# # print(mask_pred['results']['video_test_0000045'][1])

# for i in range(len(clean_pred['results']['video_test_0000045'])):
#     if clean_pred['results']['video_test_0000045'][i]['label'] == anno_data['database']['video_test_0000045']['annotations'][0]['label']:
#         clean_list.append(clean_pred['results']['video_test_0000045'][i]['segment'])
#         clean_score.append(clean_pred['results']['video_test_0000045'][i]['score'])

# for i in range(len(mask_pred['results']['video_test_0000045'])):
#     if mask_pred['results']['video_test_0000045'][i]['label'] == anno_data['database']['video_test_0000045']['annotations'][0]['label']:
#         mask_list.append(mask_pred['results']['video_test_0000045'][i]['segment'])            
#         mask_score.append(mask_pred['results']['video_test_0000045'][i]['score'])

# print(sorted(clean_list), '\n')
# print(sorted(mask_list))

# print(sorted(clean_score)[-1])
# print(sorted(mask_score)[-1])

# for i in range(len(clean_pred['results']['video_test_0000045'])):
#     if clean_pred['results']['video_test_0000045'][i]['score'] == sorted(clean_score)[-1]:
#         print(clean_pred['results']['video_test_0000045'][i]['segment'])

# for i in range(len(mask_pred['results']['video_test_0000045'])):
#     if mask_pred['results']['video_test_0000045'][i]['score'] == sorted(mask_score)[-1]:
#         print(mask_pred['results']['video_test_0000045'][i]['segment'])