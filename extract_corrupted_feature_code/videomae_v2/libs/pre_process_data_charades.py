import csv
import os
import numpy as np
import json
from dataset.loader import get_video_loader
def load_json(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def pre_process_annotations_data(cfg,data_path):
    video_seg = {}
    process_frame = {}
    process_feature = {}
    max_frame = {}
    max_feature = {}
    video_loader = get_video_loader()
    fps = cfg['chara']['fps']
    if cfg['process_configs']['dataset_type'] == 'train':
        annotations_input_file_root = cfg['file_root']['annotations_input_file_root_train']
    elif cfg['process_configs']['dataset_type'] == 'test':
        annotations_input_file_root = cfg['file_root']['annotations_input_file_root_test']

    #with open(annotations_input_file_root, 'r') as file:
    file = load_json(annotations_input_file_root)
    count = 0
    old_index = None
    for line in file:
        # new_index, start_sec, rest = line.strip().split(" ", 2)
        # end_sec, descrption = rest.strip().split("##", 1)
        # new_index, duration, timestamps, descrption = line
        # start_sec, end_sec = timestamps[0], timestamps[1]
        # video_path = os.path.join(data_path, new_index + '.mp4')
        # vr = video_loader(video_path)
        seg_start_t,seg_end_t,new_index = int(line['feature_start']+0.5), int(line['feature_end']+0.5), line['video']
        if new_index == old_index:
            count += 1
        else:
            count = 0
        old_index = new_index
        process_frame[f'{new_index}_{count}'] = set()
        video_seg[f'{new_index}_{count}'] = []
        max_frame[f'{new_index}_{count}'] = 0
        # #计算动作起始帧，要改成 int(start_sec / duration * len(video_frames) + 0.5) 
        # seg_start_t = int(start_sec / duration * len(vr) + 0.5)
        # #计算动作终止帧，要改成 int(end_sec / duration * len(video_frames) + 0.5)
        # seg_end_t = int(end_sec / duration * len(vr) + 0.5)

        if cfg['process_configs']['sample_way'] == 'random_in_action_and_background' or cfg['process_configs']['sample_way'] == 'random_continuous':
            video_seg[f'{new_index}_{count}'].append([seg_start_t, seg_end_t])

        elif cfg['process_configs']['sample_way'] == 'continuous_in_action':
            seg_middle  = int((seg_start_t + seg_end_t) / 2 + 0.5) # unit is frame
            seg_dur = seg_end_t - seg_start_t
            if cfg['process_configs']['add_noise_unity'] == 'frame':
                process_frame_start_idx = int(seg_middle - cfg['process_configs']['noise_length'] / 2 + 0.5)
                process_frame_end_idx = int(seg_middle + cfg['process_configs']['noise_length'] / 2 - 0.5)
            elif cfg['process_configs']['add_noise_unity'] == 'percentage':
                process_frame_start_idx = int(seg_middle - cfg['process_configs']['noise_length'] * seg_dur / 200 + 0.5)
                process_frame_end_idx = int(seg_middle + cfg['process_configs']['noise_length'] * seg_dur / 200 + 0.5 - 1)

            add_noise_list = [(i-1) for i in range(process_frame_start_idx, process_frame_end_idx+1)] # idx start from 1 --> from 0

            process_frame[f'{new_index}_{count}'].update(add_noise_list)

    if cfg['process_configs']['sample_way'] == 'random_in_action_and_background' or cfg['process_configs']['sample_way'] == 'random_continuous':
        for vid in video_seg:
            video_seg[vid] = sorted(video_seg[vid], key=lambda x: x[0])
            for idx in range(len(video_seg[vid])):
                if idx == 0:
                    process_start_f = 0 # frame idx 
                    process_end_f = video_seg[vid][idx][1]
                else:
                    process_start_f = video_seg[vid][idx-1][1]
                    process_end_f = video_seg[vid][idx][1]
                    if process_start_f >= process_end_f:
                        process_start_f = video_seg[vid][idx-1][0]
                        process_end_f = video_seg[vid][idx-1][1]
                        
                process_start_f = max(int(process_start_f + 0.5), 0)
                process_end_f = min(int(process_end_f + 0.5), max_frame[vid] - 1) # idx strat from 0

                if cfg['process_configs']['add_noise_unity']=='percentage':
                    length = int((video_seg[vid][idx][1] - video_seg[vid][idx][0]) * cfg['process_configs']['noise_length'] / 100 + 0.5)
                elif cfg['process_configs']['add_noise_unity']=='frame':
                    length = cfg['process_configs']['noise_length']

                # edge process, cause by error annotations
                if process_start_f >= process_end_f - length:
                    process_start_f = process_end_f - length - 1
                    if process_start_f < 0:
                        process_start_f = 0
                        process_end_f = 1 + length

                
                if cfg['process_configs']['sample_way'] == 'random_continuous':
                    # if process_start_f + 1 == process_end_f - length:
                    #     noise_position = process_start_f + 1
                    # else:
                    noise_position = np.random.randint(process_start_f, process_end_f - length + 1, 1)
                    add_noise_list = [i for i in range(int(noise_position), int(noise_position) + length)]
                    if add_noise_list[-1] + 1 > max_frame[vid]:
                        raise ValueError
                else:
                    add_noise_list = np.random.randint(process_start_f, process_end_f , cfg['process_configs']['noise_length'])

                process_frame[vid].update(add_noise_list) # here, frame idx start from 0, not 1!!!
                
    annotations_output_file = os.path.join(cfg['file_root']['output_file_dir'], 'annotations')

    # for vid in process_frame:
    #     if vid not in process_feature:
    #         process_feature[vid] = set()
    #     for frame in process_frame[vid]:
    #         clip_idx = frame // cfg['frequency']
    #         # clip_idx = min(clip_idx, max_feature[vid])
    #         process_feature[vid].add(clip_idx)
    #     process_feature[vid] = sorted(process_feature[vid]) # start from 0

    # os.makedirs(annotations_output_file, exist_ok=True)
    # if os.path.exists(os.path.join(annotations_output_file, 'processed_frames.json') and os.path.join(annotations_output_file, 'processed_features.json')):
    #     with open(os.path.join(annotations_output_file, 'processed_frames.json'), 'r') as f:
    #         process_frame = json.load(f)
    #     with open(os.path.join(annotations_output_file, 'processed_features.json'), 'r') as f:
    #         process_feature = json.load(f)
    # elif os.path.exists(os.path.join(annotations_output_file, 'processed_frames_1.json')):
    #     file_num = 1
    #     while(os.path.exists(os.path.join(annotations_output_file, 'processed_frames_' + str(file_num) +'.json'))):
    #         with open(os.path.join(annotations_output_file, 'processed_frames_' + str(file_num) +'.json'), 'r') as f:
    #             process_fr = json.load(f)
    #         with open(os.path.join(annotations_output_file, 'processed_features_' + str(file_num) +'.json'), 'r') as f:
    #             process_feat = json.load(f)            
    #         for i in process_fr:
    #             process_frame[i] = process_fr[i]
    #         for i in process_feat:
    #             process_feature[i] = process_feat[i]
    #         file_num += 1
    # else:
    #     frames_process_json = {key: sorted(list(map(int, value))) for key, value in process_frame.items()}
    #     feat_process_json = {key: sorted(list(map(int, value))) for key, value in process_feature.items()}
    #     dictt = {}
    #     dictt_feat = {}
    #     count = 0
    #     file_num = 1
    #     for i in frames_process_json:
    #         dictt[i] = frames_process_json[i]
    #         dictt_feat[i] = feat_process_json[i]
    #         count += 1
    #         if count == 5000:
    #             with open(os.path.join(annotations_output_file, 'processed_frames_' + str(file_num) +'.json'), 'w') as f:
    #                 _ = json.dump(dictt, f, indent=4)
    #             with open(os.path.join(annotations_output_file, 'processed_features_' + str(file_num) +'.json'), 'w') as f:
    #                 _ = json.dump(dictt_feat, f, indent=4)
    #             file_num += 1
    #             count = 0
    #             dictt = {}
    #             dictt_feat = {}

    #     del frames_process_json, dictt

    return process_frame
# process_feature

def pre_process_MMN_annotations_data(cfg,data_path):
    video_seg = {}
    process_frame = {}
    process_feature = {}
    max_frame = {}
    max_feature = {}
    video_loader = get_video_loader()
    fps = cfg['chara']['fps']
    if cfg['process_configs']['dataset_type'] == 'train':
        annotations_input_file_root = cfg['file_root']['annotations_input_file_root_train']
    elif cfg['process_configs']['dataset_type'] == 'test':
        annotations_input_file_root = cfg['file_root']['annotations_input_file_root_test']

    #with open(annotations_input_file_root, 'r') as file:
    file = load_json(annotations_input_file_root)
    for vid, anno in file.items():
        # new_index, start_sec, rest = line.strip().split(" ", 2)
        # end_sec, descrption = rest.strip().split("##", 1)
        # new_index, duration, timestamps, descrption = line
        # start_sec, end_sec = timestamps[0], timestamps[1]
        video_path = os.path.join(data_path, vid + '.mp4')
        vr = video_loader(video_path)
        duration = anno['duration']
        count = 0
        for timestamp in anno['timestamps']:
            process_frame[f'{vid}_{count}'] = set()
            video_seg[f'{vid}_{count}'] = []
            max_frame[f'{vid}_{count}'] = 0
            #计算动作起始帧，要改成 int(start_sec / duration * len(video_frames) + 0.5) 
            seg_start_t = int(timestamp[0] / duration * len(vr) + 0.5)
            #计算动作终止帧，要改成 int(end_sec / duration * len(video_frames) + 0.5)
            seg_end_t = int(timestamp[1] / duration * len(vr) + 0.5)

            if cfg['process_configs']['sample_way'] == 'random_in_action_and_background' or cfg['process_configs']['sample_way'] == 'random_continuous':
                video_seg[f'{vid}_{count}'].append([seg_start_t, seg_end_t])

            elif cfg['process_configs']['sample_way'] == 'continuous_in_action':
                seg_middle  = int((seg_start_t + seg_end_t) / 2 + 0.5) # unit is frame
                seg_dur = seg_end_t - seg_start_t
                if cfg['process_configs']['add_noise_unity'] == 'frame':
                    process_frame_start_idx = int(seg_middle - cfg['process_configs']['noise_length'] / 2 + 0.5)
                    process_frame_end_idx = int(seg_middle + cfg['process_configs']['noise_length'] / 2 - 0.5)
                elif cfg['process_configs']['add_noise_unity'] == 'percentage':
                    process_frame_start_idx = int(seg_middle - cfg['process_configs']['noise_length'] * seg_dur / 200 + 0.5)
                    process_frame_end_idx = int(seg_middle + cfg['process_configs']['noise_length'] * seg_dur / 200 + 0.5 - 1)

                add_noise_list = [(i-1) for i in range(process_frame_start_idx, process_frame_end_idx+1)] # idx start from 1 --> from 0

                process_frame[f'{vid}_{count}'].update(add_noise_list)
            count += 1
    annotations_output_file = os.path.join(cfg['file_root']['output_file_dir'], 'annotations')
    return process_frame

def pre_process_momentdiff_annotations_data(cfg,data_path):
    video_seg = {}
    process_frame = {}
    process_feature = {}
    max_frame = {}
    max_feature = {}
    video_loader = get_video_loader()
    fps = cfg['chara']['fps']
    if cfg['process_configs']['dataset_type'] == 'train':
        annotations_input_file_root = cfg['file_root']['annotations_input_file_root_train']
    elif cfg['process_configs']['dataset_type'] == 'test':
        annotations_input_file_root = cfg['file_root']['annotations_input_file_root_test']
    count = 0
    old_vid = None
    with open(annotations_input_file_root, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # new_index, start_sec, rest = line.strip().split(" ", 2)
            # end_sec, descrption = rest.strip().split("##", 1)
            # new_index, duration, timestamps, descrption = line
            start_sec, end_sec = data['relevant_windows'][0][0], data['relevant_windows'][0][1]
            duration = data['duration']
            vid = data['vid']
            video_path = os.path.join(data_path, vid + '.mp4')
            vr = video_loader(video_path)
            if vid == old_vid:
                count += 1
            else:
                count = 0
            old_vid = vid
            process_frame[f'{vid}_{count}'] = set()
            video_seg[f'{vid}_{count}'] = []
            max_frame[f'{vid}_{count}'] = 0
            #计算动作起始帧，要改成 int(start_sec / duration * len(video_frames) + 0.5) 
            seg_start_t = int(start_sec / duration * len(vr) + 0.5)
            #计算动作终止帧，要改成 int(end_sec / duration * len(video_frames) + 0.5)
            seg_end_t = int(end_sec / duration * len(vr) + 0.5)

            if cfg['process_configs']['sample_way'] == 'random_in_action_and_background' or cfg['process_configs']['sample_way'] == 'random_continuous':
                video_seg[f'{vid}_{count}'].append([seg_start_t, seg_end_t])

            elif cfg['process_configs']['sample_way'] == 'continuous_in_action':
                seg_middle  = int((seg_start_t + seg_end_t) / 2 + 0.5) # unit is frame
                seg_dur = seg_end_t - seg_start_t
                if cfg['process_configs']['add_noise_unity'] == 'frame':
                    process_frame_start_idx = int(seg_middle - cfg['process_configs']['noise_length'] / 2 + 0.5)
                    process_frame_end_idx = int(seg_middle + cfg['process_configs']['noise_length'] / 2 - 0.5)
                elif cfg['process_configs']['add_noise_unity'] == 'percentage':
                    process_frame_start_idx = int(seg_middle - cfg['process_configs']['noise_length'] * seg_dur / 200 + 0.5)
                    process_frame_end_idx = int(seg_middle + cfg['process_configs']['noise_length'] * seg_dur / 200 + 0.5 - 1)

                add_noise_list = [(i-1) for i in range(process_frame_start_idx, process_frame_end_idx+1)] # idx start from 1 --> from 0

                process_frame[f'{vid}_{count}'].update(add_noise_list)
    annotations_output_file = os.path.join(cfg['file_root']['output_file_dir'], 'annotations')
    return process_frame