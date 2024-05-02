"""Extract features for temporal action detection datasets"""
##thumos videomae
import argparse
import os
import random
import json
import numpy as np
import torch
import yaml
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm
import cv2

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader
from libs.pre_process_data_thumos import pre_process_annotations_data
from libs.cutout import fuzzy_cutout_fn
from libs.light import add_light
from libs.motion_blur import motion_blur
from libs.cover import cover

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)

class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='THUMOS14',
        choices=['THUMOS14', 'FINEACTION'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default='YOUR_PATH/thumos14_video',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default='YOUR_PATH/thumos14_video/th14_vit_g_16_4',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='vit_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='YOUR_PATH/vit_g_hyrbid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):
    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    else:
        raise NotImplementedError()

def extract_feature(cfg):
    # preparation
    print('preparing..', end='   ')
    save_path = os.path.join(cfg['file_root']['output_file_dir'], 'features')
    dataset_type = cfg['process_configs']['dataset_type']
    if dataset_type == 'test':
        data_path = cfg['file_root']['video_input_dir_test']
    elif dataset_type == 'val':
        data_path = cfg['file_root']['video_input_dir_val']
    frequency = cfg['frequency']
    ckpt_path = cfg['file_root']['ckpt_path']
    with open('libs/tridet_feature_max.json', 'r') as file:
        max_data = json.load(file)
    cover_img = cv2.imread(os.path.join(".", "libs", "cover_img", "hand.png"), -1)
    video_loader = get_video_loader()
    start_idx_range = get_start_idx_range('THUMOS14')
    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])
    chunk_size = cfg['chunk_size']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # feature_to_frame = pre_process_annotations_data(cfg)
    frames_process, feature_to_frame = pre_process_annotations_data(cfg)
    with open(os.path.join(cfg['file_root']['output_file_dir'],'annotations','frame_process.json'),'w') as file: 
        frames_process_json = {key[:-4]: sorted(list(map(int, value))) for key, value in frames_process.items()}
        _ = json.dump(frames_process_json, file, indent=4, sort_keys=True) 
    with open(os.path.join(cfg['file_root']['output_file_dir'],'annotations','feature_process.json'),'w') as file: 
        feature_process_json = {key[:-4]: sorted(list(map(int, value))) for key, value in feature_to_frame.items()}
        _ = json.dump(feature_process_json, file, indent=4, sort_keys=False) 
    del frames_process_json, feature_process_json

    # get video path
    vid_list = os.listdir(data_path)
    random.shuffle(vid_list)
    # vid_list.sort()
    print('Done')

    # get model & load ckpt
    print('create model..', end='   ')
    model = create_model(
        'vit_giant_patch14_224',
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=chunk_size,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()
    print('Done')

    # extract feature
    print('start extract..')
    num_videos = len(vid_list)

    for idx, vid_name in tqdm(enumerate(vid_list)):
        if cfg['process_configs']['process_type'] == 'replace':
            url = os.path.join(save_path, vid_name.split('.')[0] + '.npy')
            video_path = os.path.join(data_path, vid_name)
            vr = video_loader(video_path)
            feature_old = np.load(os.path.join(cfg['file_root']['feature_input_dir'], vid_name.split('.')[0] + '.npy'))

            for start_idx in tqdm(feature_to_frame[vid_name]):
                get_frames_idx = np.arange(start_idx * frequency, start_idx * frequency + 16)
                data = vr.get_batch(get_frames_idx).asnumpy()

                for frame_idx in get_frames_idx:
                    if frame_idx in frames_process[vid_name]:
                        if cfg['process_configs']['process_way'] == 'mask':
                            data[frame_idx - start_idx * frequency] = np.zeros(data[frame_idx - start_idx * frequency].shape)
                        elif cfg['process_configs']['process_way'] == 'cutout':
                            process_idx = frame_idx - start_idx*frequency
                            process_pre_idx = max(0, frame_idx-30)
                            process_lat_idx = min(max_data[vid_name.split('.')[0]][0]-1, frame_idx+30)
                            data_process = vr.get_batch(np.array([process_pre_idx, process_lat_idx])).asnumpy()
                            data[process_idx] = fuzzy_cutout_fn(data_process[0], data[process_idx], data_process[1])
                        elif cfg['process_configs']['process_way'] == 'light':
                            data[frame_idx - start_idx * frequency] = add_light(data[frame_idx - start_idx * frequency])
                        elif cfg['process_configs']['process_way'] == 'motion_blur':
                            data[frame_idx - start_idx * frequency] = motion_blur(data[frame_idx - start_idx * frequency])
                        elif cfg['process_configs']['process_way'] == 'cover':
                            data[frame_idx - start_idx * frequency] = cover(data[frame_idx - start_idx * frequency], cover_img)
                        

                frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
                frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
                input_data = frame_q.unsqueeze(0).cuda()

                with torch.no_grad():
                    feature_replace = model.forward_features(input_data).cpu().numpy()
                    feature_old[start_idx] = feature_replace

            # [N, C]
            np.save(url, feature_old)
            print(f'[{idx} / {num_videos}]: save feature on {url}')
            
        elif cfg['process_configs']['process_type'] == 'keep':
            feat_ori = []
            url = os.path.join(save_path, vid_name.split('.')[0] + '.npy')
            if os.path.exists(url):
                continue
            video_path = os.path.join(data_path, vid_name)
            vr = video_loader(video_path)
            frame_cnt = len(vr)
            feature_list = []
            for feature_idx in range((frame_cnt - chunk_size) // frequency):
                get_frames_idx = np.arange(feature_idx * frequency, feature_idx * frequency + chunk_size)
                data = vr.get_batch(get_frames_idx).asnumpy()
                
                frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
                frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
                input_data = frame_q.unsqueeze(0).cuda()
                
                with torch.no_grad():
                    feature = model.forward_features(input_data)
                    feature_list.append(feature.cpu().numpy())

            np.save(url, np.vstack(feature_list))
            print(f'[{idx} / {num_videos}]: save feature on {url}')

def load_config(config_file='./configs/project_configs_thumos_mae.yaml'):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config

def check_configs(cfg):
    assert (cfg['process_configs']['sample_way'] in ['continuous_in_action_and_background', 'continuous_in_action', 'continuous_in_background',
                                                     'random_in_action', 'random_in_background', 'random_in_action_and_background'])

    assert (cfg['process_configs']['add_noise_unity'] in ['frame', 'percentage'])
    assert (cfg['process_configs']['process_way'] in ['mask', 'mixup', 'nothing', 'cutout', 'light', 'motion_blur', 'cover'])

    if not os.path.exists(cfg['file_root']['output_file_dir']):
        annotations_file = os.path.join(cfg['file_root']['output_file_dir'], 'annotations')
        feature_file = os.path.join(cfg['file_root']['output_file_dir'], 'i3d_features')
        os.makedirs(annotations_file)
        os.makedirs(feature_file)

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg = load_config()
    check_configs(cfg)
    extract_feature(cfg)