import os
import torch
import yaml
import json
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import cv2

# local code
from models.i3d.pytorch_i3d import InceptionI3d
from models.i3d.i3dpt import I3D
from libs.pre_process_data import pre_process_annotations_data
from libs.cutout import fuzzy_cutout_fn
from libs.light import add_light
from libs.motion_blur import motion_blur
from libs.cover import cover

def load_frame(frame_file, resize=False):

    data = Image.open(frame_file)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)

    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data

def oversample_data(data): # (39, 16, 224, 224, 2)  # Check twice

    data_flip = np.array(data[:,:,:,::-1,:])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])   # ,:,16:240,58:282,:
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
        data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

def load_rgb_batch(frames_dir, rgb_files,  
                   frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            if frame_indices[i][j] >= len(rgb_files):
                batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, 
                    rgb_files[-1]), resize)
            else:
                batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, 
                    rgb_files[frame_indices[i][j]]), resize)

    return batch_data   

def load_config(config_file='./configs/project_configs.yaml'):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config

def extract_feature(cfg):
    """0. init"""
    print('strat to init...', end='   ')
    random.seed(123)
    np.random.seed(123)
    mode = cfg['extract_feat_args']['mode'] 
    load_model=cfg['model'][cfg['model']['model_name']]['load_model']
    sample_mode=cfg['model'][cfg['model']['model_name']]['sample_mode']
    feat_input_dir=cfg['model'][cfg['model']['model_name']]['feat_input_dir']
    feat_output_dir=cfg['file_root']['output_file_dir']
    batch_size=cfg['model'][cfg['model']['model_name']]['batch_size']
    frequency=cfg['model'][cfg['model']['model_name']]['i3d_frequency']
    chunk_size=cfg['model'][cfg['model']['model_name']]['i3d_chunk_size']
    dataset_type = cfg['process_configs']['dataset_type']
    with open('./actionformer_feature_max_all.json', 'r') as file:
        max_data = json.load(file)
    if dataset_type == 'test':
        img_input_dir = cfg['file_root']['test_img_input_dir']
    else:
        img_input_dir = cfg['file_root']['val_img_input_dir']
    require_resize = (sample_mode == 'resize')
    cover_img = cv2.imread(os.path.join(".", "libs", "cover_img", "hand.png"), -1)
    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])    
    count = 0
    print("Done")

    """1. count the range of process frames and feature"""
    print('strat to preprocess the annotations data...', end='   ')

    frames_process, feature_process = pre_process_annotations_data(cfg)
    with open(os.path.join(cfg['file_root']['output_file_dir'],'annotations','frame_process_'+dataset_type+'.json'),'w') as file: 
        frames_process_json = {key: sorted(list(map(int, value))) for key, value in frames_process.items()}
        _ = json.dump(frames_process_json, file, indent=4, sort_keys=True) 
    with open(os.path.join(cfg['file_root']['output_file_dir'],'annotations','feature_process_'+dataset_type+'.json'),'w') as file: 
        feature_process_json = {key: sorted(list(map(int, value))) for key, value in feature_process.items()}
        _ = json.dump(feature_process_json, file, indent=4, sort_keys=False) 
    print("Done")
    del frames_process_json, feature_process_json

    """2. setup the model""" 
    print('strat to setup the model...', end='   ')

    if cfg['model']['model_name'] == 'pgcn':
        i3d = I3D(num_classes=400, modality='rgb')
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    i3d.eval()
    print("Done")

    """3. extract and replace feature"""
    print('strat to extract and replace feature...')
    if not os.path.exists(os.path.join(feat_output_dir, 'features')):
        os.makedirs(os.path.join(feat_output_dir, 'features'))
    for video_name in feature_process:
        if cfg['process_configs']['process_type'] == 'replace':
            url = os.path.join(feat_output_dir, 'features', video_name+'.npy')
            if os.path.exists(url):
                continue
            # per video
            if not os.path.exists(os.path.join(feat_input_dir, '{}.npy'.format(video_name))):
                continue
            feat_ori_root = os.path.join(feat_input_dir, '{}.npy'.format(video_name))
            feat_ori = np.load(feat_ori_root)
            frames_dir = os.path.join(img_input_dir, video_name)
            if mode == 'rgb':
                rgb_files = [i for i in os.listdir(frames_dir) if i.startswith('img')]
                rgb_files.sort()
                frame_cnt = len(rgb_files)

            assert(frame_cnt > chunk_size)
            
            for feature_idx in feature_process[video_name]: 
                # per feature
                print(f'feature idx is: {feature_idx}')
                frequency = len(rgb_files) // len(feat_ori)
                feature_frame_indices = []
                feature_frame_indices.append([j for j in range(feature_idx * frequency, feature_idx * frequency + chunk_size)])
                feature_frame_indices = np.array(feature_frame_indices) # shape is (chunk_num, chunk_size)
                chunk_num = feature_frame_indices.shape[0]
                batch_num = int(np.ceil(chunk_num / batch_size))
                feature_frame_indices = np.array_split(feature_frame_indices, batch_num, axis=0)
                
                if sample_mode == 'oversample':
                    full_features = [[] for _ in range(10)]
                else:
                    full_features = [[]]
                
                for batch_id in range(batch_num): 
                    batch_data = load_rgb_batch(frames_dir, rgb_files,  # load rgb image
                                                feature_frame_indices[batch_id], require_resize)
                    
                    for chunk_idx in range(batch_data.shape[0]): # process specified image
                        for frame_idx in range(chunk_size):
                            if feature_frame_indices[batch_id][chunk_idx][frame_idx] in frames_process[video_name]:
                                if cfg['process_configs']['process_way'] == 'mask':
                                    batch_data[chunk_idx][frame_idx] = -1 * np.ones_like(batch_data[chunk_idx][frame_idx])
                                elif cfg['process_configs']['process_way'] == 'mixup':
                                    mixup_img_name = random.choice(rgb_files)
                                    mixup_img_data = load_frame(os.path.join(frames_dir, mixup_img_name))                                
                                    if cfg['process_configs']['mixup_weight_is_fixed']:
                                        mixup_weight = cfg['process_configs']['mixup_weight']
                                    else:
                                        mixup_weight = np.random.beta(0.5, 0.5)
                                    batch_data[chunk_idx][frame_idx] = batch_data[chunk_idx][frame_idx] * mixup_weight  + mixup_img_data * (1. - mixup_weight)
                                elif cfg['process_configs']['process_way'] == 'cutout':
                                    process_idx = feature_frame_indices[batch_id][chunk_idx][frame_idx]
                                    process_pre_idx = max(0, process_idx-30)
                                    process_lat_idx = min(max_data[video_name][0]-1, process_idx+30)
                                    data_process = []
                                    data_process.append(load_frame(os.path.join(frames_dir, rgb_files[process_pre_idx]), resize=False))
                                    data_process.append(load_frame(os.path.join(frames_dir, rgb_files[process_lat_idx]), resize=False))
                                    batch_data[chunk_idx][frame_idx] = fuzzy_cutout_fn(data_process[0], batch_data[chunk_idx][frame_idx], data_process[1])
                                elif cfg['process_configs']['process_way'] == 'light':
                                    batch_data[chunk_idx][frame_idx] = add_light(batch_data[chunk_idx][frame_idx])
                                elif cfg['process_configs']['process_way'] == 'motion_blur':
                                    batch_data[chunk_idx][frame_idx] = motion_blur(batch_data[chunk_idx][frame_idx])
                                elif cfg['process_configs']['process_way'] == 'cover':
                                    batch_data[chunk_idx][frame_idx] = cover(batch_data[chunk_idx][frame_idx], cover_img)
                                elif cfg['process_configs']['process_way'] == 'nothing':
                                    pass

                    if sample_mode == 'oversample':
                        batch_data_ten_crop = oversample_data(batch_data)
                        for i in range(10): # oversample require 10 times
                            assert(batch_data_ten_crop[i].shape[-2]==224)
                            assert(batch_data_ten_crop[i].shape[-3]==224)
                            full_features[i].append(i3d.forward_batch(batch_data_ten_crop[i]))

                    elif sample_mode == 'center_crop':
                        batch_data = batch_data[:,:,16:240,58:282,:]
                        full_features[0].append(i3d.forward_batch(batch_data))
                
                full_features = [np.concatenate(i, axis=0) for i in full_features]
                full_features = [np.expand_dims(i, axis=0) for i in full_features]
                full_features = np.concatenate(full_features, axis=0)
                full_features = np.mean(full_features, axis=0)
                
                feat_ori[feature_idx - 1] = full_features # replace feature
            
        elif cfg['process_configs']['process_type'] == 'keep':
            url = os.path.join(feat_output_dir, 'features', video_name+'.npy')
            if os.path.exists(url):
                continue
            feat_ori = []
            frames_dir = os.path.join(img_input_dir, video_name)
            if mode == 'rgb':
                rgb_files = [i for i in os.listdir(frames_dir) if i.startswith('img') and i.endswith('.jpg')]
                rgb_files.sort()
                frame_cnt = len(rgb_files)

            for feature_idx in range((frame_cnt - chunk_size) // frequency): 
                print(f'feature idx is: {feature_idx}')
                feature_frame_indices = []
                feature_frame_indices.append([j for j in range(feature_idx * frequency, feature_idx * frequency + chunk_size)])
                feature_frame_indices = np.array(feature_frame_indices) # shape is (chunk_num, chunk_size)
                chunk_num = feature_frame_indices.shape[0]
                batch_num = int(np.ceil(chunk_num / batch_size))
                feature_frame_indices = np.array_split(feature_frame_indices, batch_num, axis=0)
            
                if sample_mode == 'oversample':
                    full_features = [[] for _ in range(10)]
                else:
                    full_features = [[]]
                
                for batch_id in range(batch_num): 
                    batch_data = load_rgb_batch(frames_dir, rgb_files,  # load rgb image
                                                feature_frame_indices[batch_id], require_resize)
                    
                    if sample_mode == 'oversample':
                        batch_data_ten_crop = oversample_data(batch_data)
                        for i in range(10): # oversample require 10 times
                            assert(batch_data_ten_crop[i].shape[-2]==224)
                            assert(batch_data_ten_crop[i].shape[-3]==224)
                            full_features[i].append(i3d.forward_batch(batch_data_ten_crop[i]))

                    elif sample_mode == 'center_crop':
                        batch_data = batch_data[:,:,16:240,58:282,:]
                        full_features[0].append(i3d.forward_batch(batch_data))
                
                full_features = [np.concatenate(i, axis=0) for i in full_features]
                full_features = [np.expand_dims(i, axis=0) for i in full_features]
                full_features = np.concatenate(full_features, axis=0)
                full_features = np.mean(full_features, axis=0)
                
                feat_ori.append(full_features)            
            feat_ori = np.squeeze(feat_ori)
        np.save(url, feat_ori)
        print(f'{video_name} done\nframe cnt are: {frame_cnt}\nfeature shape is: {len(feat_ori)}')

        # count += 1
        # print(f'{video_name} done:\nreplace times is: {len(feature_process[video_name])}, video idx is: {count}/{len(feature_process)}')



def check_configs(cfg):
    assert (cfg['process_configs']['sample_way'] in ['interval_in_action',  
                                                     'random_in_action', 'random_in_background', 'random_in_action_and_background',
                                                     'continuous_in_action', 'continuous_in_background', 'continuous_in_action_and_background'])
    assert (cfg['process_configs']['noise_length_type'] in ['variable', 'fixed'])
    assert (cfg['process_configs']['noise_length_min'] >= 0)
    assert (cfg['process_configs']['add_noise_unity'] in ['frame', 'percentage'])
    assert (cfg['process_configs']['rgb_only'] == True),"only support rgb now"
    assert (cfg['extract_feat_args']['mode'] == 'rgb'),"only support rgb now"
    assert (cfg['process_configs']['noise_position_at_segment_type'] in ['variable', 'fixed'])
    assert (cfg['process_configs']['process_way'] in ['mask', 'mixup', 'nothing', 'cutout', 'light', 'motion_blur', 'cover'])
    assert (cfg['process_configs']['mixup_weight'] >= 0 and cfg['process_configs']['mixup_weight'] <= 1)

    if cfg['process_configs']['noise_position_at_segment_type'] == 'variable':
        assert (cfg['process_configs']['noise_length_type'] == 'fixed'), "only support fixed length when position is variable"
        
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
