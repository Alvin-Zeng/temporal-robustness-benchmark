# -*- coding:gb2312 -*-
import json
import os
import copy
import numpy as np

def read_feature_frame_max(cfg, data):
    chunk_size = cfg['model'][cfg['model']['model_name']]['i3d_chunk_size']
    frequency = cfg['model'][cfg['model']['model_name']]['i3d_frequency']
    ori_feature_dir = cfg['model'][cfg['model']['model_name']]['feat_input_dir']
    # if not os.path.exists(os.path.join(feat_input_dir, '{}.npy'.format(video_name))):
    feature_and_frame_max_dict = {}
    # if not os.path.exists('./'+cfg['model']['model_name']+'_feature_max_all.json'):
    if cfg['process_configs']['process_type'] == 'replace':
        if cfg['process_configs']['dataset_type'] == 'test':
            for video_index_str in data['database']:
                if video_index_str[6] == 't':
                    img_file_list = os.listdir(os.path.join(cfg['file_root']['test_img_input_dir'], video_index_str))
                    img_file_list.sort()
                    img_max = int(img_file_list[-1][4:9])
                    if not os.path.exists(os.path.join(ori_feature_dir, '{}.npy'.format(video_index_str))):
                        continue
                    feature = np.load(os.path.join(ori_feature_dir, '{}.npy'.format(video_index_str)))
                    feature_max = len(feature)
            # while feature_max * frequency + chunk_size + 1 > img_max:
            #     feature_max -= 1
                feature_and_frame_max_dict[video_index_str] = [img_max, feature_max]
        elif cfg['process_configs']['dataset_type'] == 'val':
            for video_index_str in data['database']:
                if video_index_str[6] == 'v':
                    img_file_list = os.listdir(os.path.join(cfg['file_root']['val_img_input_dir'], video_index_str))
                    img_file_list.sort()
                    img_max = int(img_file_list[-1][4:9])
            # feature_max = (img_max - 1 - chunk_size) // frequency + 1 
                    if not os.path.exists(os.path.join(ori_feature_dir, '{}.npy'.format(video_index_str))):
                        continue
                    feature = np.load(os.path.join(ori_feature_dir, '{}.npy'.format(video_index_str)))
                    feature_max = len(feature)
            # while feature_max * frequency + chunk_size + 1 > img_max:
            #     feature_max -= 1
                    feature_and_frame_max_dict[video_index_str] = [img_max, feature_max]
    elif cfg['process_configs']['process_type'] == 'keep':
        if cfg['process_configs']['dataset_type'] == 'test':
            for video_index_str in data['database']:
                if video_index_str[6] == 't':
                    img_file_list = os.listdir(os.path.join(cfg['file_root']['test_img_input_dir'], video_index_str))
                    img_file_list.sort()
                    img_max = int(img_file_list[-1][4:9])
                    feature_max = (img_max - chunk_size) // frequency + 1

                feature_and_frame_max_dict[video_index_str] = [img_max, feature_max]
        elif cfg['process_configs']['dataset_type'] == 'val':
            for video_index_str in data['database']:
                if video_index_str[6] == 'v':
                    img_file_list = os.listdir(os.path.join(cfg['file_root']['val_img_input_dir'], video_index_str))
                    img_file_list.sort()
                    img_max = int(img_file_list[-1][4:9])
                    feature_max = (img_max - chunk_size) // frequency + 1
            # while feature_max * frequency + chunk_size + 1 > img_max:
            #     feature_max -= 1
                    feature_and_frame_max_dict[video_index_str] = [img_max, feature_max]

    return feature_and_frame_max_dict

def pre_process_annotations_data(cfg):   
    chunk_size = cfg['model'][cfg['model']['model_name']]['i3d_chunk_size']
    # frequency = cfg['model'][cfg['model']['model_name']]['i3d_frequency']
    with open(cfg['file_root']['annotations_input_file_root'],'r') as f: 
        data = json.load(f)

    data_copy = copy.deepcopy(data)
    feature_process = {} 
    frames_process = {} 
    feature_and_frame_max_dict = read_feature_frame_max(cfg, data)

    length_threshold = cfg['process_configs']['process_annotations_length_threshold']

    if length_threshold == 0 and cfg['process_configs']['dataset_type'] == 'test':
        feature_process['video_test_0001292'] = []
        
    # feature_file_list = [feature_name[:-4] for feature_name in os.listdir(os.path.join(cfg['file_root']['output_file_dir'], 'i3d_features'))]

    for video_index_str in data['database']:
        # if video_index_str in feature_file_list:
        #     continue
        if video_index_str[6] == cfg['process_configs']['dataset_type'][0]:
            frame_max, feature_max = feature_and_frame_max_dict[video_index_str]
            frequency = frame_max // feature_max
            time = data_copy['database'][video_index_str].get('duration')
            annotations_list_sorted = sorted(data_copy['database'][video_index_str]['annotations'],key = lambda x:x['segment(frames)'][1]) # ????????????
            
            for j in range(len(annotations_list_sorted)):
                if annotations_list_sorted[j]['segment'][1] > time:
                    del annotations_list_sorted[j:]
                    break
            
            data_copy['database'][video_index_str]['annotations'] = annotations_list_sorted
            annotations_length = len(annotations_list_sorted)
            if annotations_length >= length_threshold:   
                annotations_list_sorted = sorted(data_copy['database'][video_index_str]['annotations'],key = lambda x:x['segment(frames)'][0]) # ????????????
                all_segment_frames = [] 
                
                if cfg['process_configs']['add_noise_to_all_action']:
                    for annotations_idx in range(annotations_length):
                        all_segment_frames.append(annotations_list_sorted[annotations_idx]['segment(frames)'])

                else: 
                    segment_index = int(cfg['process_configs']['noise_position_at_video'] / 100 * (annotations_length-1) + 0.5) # ????????????
                    
                    if length_threshold>=4:

                        if segment_index <= 0: 
                            segment_index = 1
                        elif segment_index >= annotations_length - 1: 
                            segment_index = annotations_length - 2
                    all_segment_frames.append(annotations_list_sorted[segment_index]['segment(frames)']) 
                    
                feature_process[video_index_str] = set() 
                frames_process[video_index_str] = set()  
                if cfg['process_configs']['noise_length'] != 0:
                    for segment_idx in range(len(all_segment_frames)):
                        if cfg['process_configs']['sample_way'] == 'continuous_in_action':
                            frame1, frame2 = all_segment_frames[segment_idx]
                            frame_position = frame1 + (frame2 - frame1) * cfg['process_configs']['noise_position_at_segment'] / 100
                            if cfg['process_configs']['add_noise_unity'] == 'frame':
                                frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                                frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                            elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                                frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame2 - frame1) / 200
                                frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame2 - frame1) / 200

                            frame_start = int(frame_start + 0.5)
                            frame_end = int(frame_end + 0.5)
                            add_noise_list = [j for j in range(frame_start, frame_end + 1)]
                            frames_process[video_index_str].update(add_noise_list) 

                            feature_start = max((frame_start - 1 - chunk_size) // frequency + 1, 0)
                            feature_end = max((frame_end - 1) // frequency, 0)
                            feature_end = min(feature_max, feature_end)
                            change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                            feature_process[video_index_str].update(change_feature_list)
                            
                        elif cfg['process_configs']['sample_way'] == 'interval_in_action':
                            assert (cfg['process_configs']['add_noise_unity'] == 'frame')
                            frame1, frame2 = all_segment_frames[segment_idx]
                            add_noise_list = np.linspace(frame1, frame2, cfg['process_configs']['noise_length'], dtype=int)
                            frames_process[video_index_str].update(add_noise_list)

                            feature_start = max((add_noise_list[0] - 1 - chunk_size) // frequency + 1, 0)
                            feature_end = max((add_noise_list[-1] - 1) // frequency, 0)
                            feature_end = min(feature_max, feature_end)
                            change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                            feature_process[video_index_str].update(change_feature_list)
                            
                        elif cfg['process_configs']['sample_way'] == 'random_in_action':
                            frame1, frame2 = all_segment_frames[segment_idx]
                            add_noise_list = np.random.randint(frame1, frame2, cfg['process_configs']['noise_length'])
                            frames_process[video_index_str].update(add_noise_list)
                            for sample_img in add_noise_list:
                                feature_start = max((sample_img - 1 - chunk_size) // frequency + 1, 0)
                                feature_end = min(max((sample_img - 1) // frequency, 0), feature_max)
                                change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                                feature_process[video_index_str].update(change_feature_list)
                        
                        elif cfg['process_configs']['sample_way'] == 'random_in_background':
                            frame_cur_start, frame_cur_end = all_segment_frames[segment_idx]
                            add_noise_list = []
                            if segment_idx != 0:
                                _, frame_last_end = all_segment_frames[segment_idx-1]
                                if frame_last_end + 1 <= frame_cur_start - 1:
                                    add_noise_list = np.random.randint(frame_last_end+1, frame_cur_start-1, cfg['process_configs']['noise_length'])

                            elif segment_idx == 0:
                                if frame_cur_start > 1:
                                    add_noise_list = np.random.randint(1, frame_cur_start - 1, cfg['process_configs']['noise_length'])

                            frames_process[video_index_str].update(add_noise_list)

                            if segment_idx == len(all_segment_frames) - 1:
                                add_noise_list = np.random.randint(frame_cur_end, frame_max, cfg['process_configs']['noise_length'])
                                frames_process[video_index_str].update(add_noise_list)

                            for sample_img in add_noise_list:
                                feature_start = max((sample_img - 1 - chunk_size) // frequency + 1, 0)
                                feature_end = min(max((sample_img - 1) // frequency, 0), feature_max)
                                change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                                feature_process[video_index_str].update(change_feature_list)
                            
                        elif cfg['process_configs']['sample_way'] == 'random_in_action_and_background':
                            frame_cur_start, frame_cur_end = all_segment_frames[segment_idx]
                            add_noise_list = []
                            if segment_idx != 0:
                                _, frame_last_end = all_segment_frames[segment_idx-1]
                                if frame_last_end + 1 <= frame_cur_end:
                                    add_noise_list = np.random.randint(frame_last_end, frame_cur_end, cfg['process_configs']['noise_length'])
                            elif segment_idx == 0:
                                if frame_cur_end > 1:
                                    add_noise_list = np.random.randint(1, frame_cur_end, cfg['process_configs']['noise_length'])
                            frames_process[video_index_str].update(add_noise_list)
                            for sample_img in add_noise_list:
                                feature_start = max((sample_img - 1 - chunk_size) // frequency + 1, 0)
                                feature_end = min(max((sample_img - 1) // frequency, 0), feature_max)
                                change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                                feature_process[video_index_str].update(change_feature_list)
                        
                        elif cfg['process_configs']['sample_way'] == 'continuous_in_background':
                            frame_cur_start, frame_cur_end = all_segment_frames[segment_idx]
                            frame_position = None
                            add_noise_list = []
                            if segment_idx != 0:
                                _, frame_last_end = all_segment_frames[segment_idx-1]
                                if frame_last_end + 1 <= frame_cur_start - 1:
                                    frame_position = frame_last_end + 1 + (frame_cur_start - frame_last_end - 2) * cfg['process_configs']['noise_position_at_segment'] / 100

                            elif segment_idx == 0:
                                if frame_cur_start > 1:
                                    frame_position = 1 + (frame_cur_start - 1 - 1) * cfg['process_configs']['noise_position_at_segment'] / 100

                            if frame_position != None:
                                if cfg['process_configs']['add_noise_unity'] == 'frame':
                                    frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                                    frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                                elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                                    frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame_cur_end - frame_cur_start) / 200
                                    frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame_cur_end - frame_cur_start) / 200
                                frame_start = max(int(frame_start + 0.5), 1)
                                frame_end = min(int(frame_end + 0.5), frame_max)
                                add_noise_list = [j for j in range(frame_start, frame_end + 1)]
                                frames_process[video_index_str].update(add_noise_list)
                                
                                for sample_img in add_noise_list:
                                    feature_start = max((sample_img - 1 - chunk_size) // frequency + 1, 0)
                                    feature_end = min(max((sample_img - 1) // frequency, 0), feature_max)
                                    change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                                    feature_process[video_index_str].update(change_feature_list)        

                            if segment_idx == len(all_segment_frames) - 1:
                                frame_position = frame_cur_end + (frame_max - frame_cur_end) * cfg['process_configs']['noise_position_at_segment'] / 100
                                if cfg['process_configs']['add_noise_unity'] == 'frame':
                                    frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                                    frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                                elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                                    frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame_cur_end - frame_cur_start) / 200
                                    frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame_cur_end - frame_cur_start) / 200
                                frame_start = int(frame_start + 0.5)
                                frame_end = min(int(frame_end + 0.5), frame_max)
                                add_noise_list = [j for j in range(frame_start, frame_end + 1)]
                                frames_process[video_index_str].update(add_noise_list)

                                for sample_img in add_noise_list:
                                    feature_start = max((sample_img - 1 - chunk_size) // frequency + 1, 0)
                                    feature_end = min(max((sample_img - 1) // frequency, 0), feature_max)
                                    change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                                    feature_process[video_index_str].update(change_feature_list)                            
                        
                        elif cfg['process_configs']['sample_way'] == 'continuous_in_action_and_background':
                            frame_cur_start, frame_cur_end = all_segment_frames[segment_idx]
                            add_noise_list = []
                            frame_position = None
                            if segment_idx != 0:
                                _, frame_last_end = all_segment_frames[segment_idx-1]
                                if frame_last_end + 1 <= frame_cur_end:
                                    select_num = np.random.randint(frame_last_end + 1, frame_cur_end, 1)
                                    if select_num[0] < frame_cur_start:
                                        frame_position = frame_last_end + 1 + (frame_cur_start - frame_last_end - 2) * cfg['process_configs']['noise_position_at_segment'] / 100
                                    else:
                                        frame_position = frame_cur_start + (frame_cur_end - frame_cur_start) * cfg['process_configs']['noise_position_at_segment'] / 100
                            elif segment_idx == 0:
                                if frame_cur_start > 1:
                                    select_num = np.random.randint(1, frame_cur_end, 1)
                                    if select_num[0] < frame_cur_start:
                                        frame_position = 1 + (frame_cur_start - 2) * cfg['process_configs']['noise_position_at_segment'] / 100
                                    else:
                                        frame_position = frame_cur_start + (frame_cur_end - frame_cur_start) * cfg['process_configs']['noise_position_at_segment'] / 100
                            if frame_position != None:
                                if cfg['process_configs']['add_noise_unity'] == 'frame':
                                    frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                                    frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                                elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                                    frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame_cur_end - frame_cur_start) / 200
                                    frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame_cur_end - frame_cur_start) / 200
                                frame_start = int(frame_start + 0.5)
                                frame_end = min(int(frame_end + 0.5), frame_max)
                                for j in range(frame_start, frame_end + 1):
                                    add_noise_list.append(j)
                                frames_process[video_index_str].update(add_noise_list)

                                for sample_img in add_noise_list:
                                    feature_start = max((sample_img - 1 - chunk_size) // frequency + 1, 0)
                                    feature_end = min(max((sample_img - 1) // frequency, 0), feature_max)
                                    change_feature_list = [j for j in range(feature_start, feature_end + 1)]
                                    feature_process[video_index_str].update(change_feature_list)
                            
            else:
                del data_copy['database'][video_index_str]
        else:
            continue
        
    return frames_process, feature_process

def mask_feature(cfg):   
    chunk_size = cfg['model'][cfg['model']['model_name']]['i3d_chunk_size']
    frequency = cfg['model'][cfg['model']['model_name']]['i3d_frequency']
    
    with open(cfg['file_root']['annotations_input_file_root'],'r') as f: 
        data = json.load(f)

    data_copy = copy.deepcopy(data)
    feature_process = {}
    frames_process = {}
    feature_max_dict = {}

    if not os.path.exists('./'+cfg['model']['model_name']+'_feature_max_all.json'):
        assert(False)
        # for video_index_str in data['database']:
        #     if video_index_str[6] == 'v':
        #         img_file_list = os.listdir(os.path.join(cfg['file_root']['img_input_dir'], video_index_str))
        #         img_file_list.sort()
        #         img_max = int(img_file_list[-1][4:9])
        #         feature_max = (img_max - 1 - chunk_size) // frequency + 1 # feature???????
        #         while feature_max * frequency + chunk_size + 1 > img_max:
        #             feature_max -= 1
        #         feature_max_dict[video_index_str] = feature_max

        # with open('./'+cfg['model']['model_name']+'_feature_max_val.json','w') as file:
        #     _ = json.dump(feature_max_dict, file, indent=4, sort_keys=False) 
    else:
        with open('./'+cfg['model']['model_name']+'_feature_max_all.json','r') as f: 
            feature_max_dict = json.load(f)

    length_threshold = cfg['process_configs']['process_annotations_length_threshold']

    if length_threshold == 0:
        feature_process['video_test_0001292'] = []
        
    # feature_file_list = [feature_name[:-4] for feature_name in os.listdir(os.path.join(cfg['file_root']['output_file_dir'], 'i3d_features'))]

    for video_index_str in data['database']: 
        # if video_index_str in feature_file_list:
        #     continue
        feature_max = feature_max_dict[video_index_str]
        # print(video_index_str)
        time = data_copy['database'][video_index_str].get('duration')
        annotations_list_sorted = sorted(data_copy['database'][video_index_str]['annotations'],key = lambda x:x['segment(frames)'][1]) # ????????????
        
        for j in range(len(annotations_list_sorted)):
            if annotations_list_sorted[j]['segment'][1] > time:
                del annotations_list_sorted[j:]
                break
            
        data_copy['database'][video_index_str]['annotations'] = annotations_list_sorted
        annotations_length = len(annotations_list_sorted) 
        if annotations_length >= length_threshold:   
            annotations_list_sorted = sorted(data_copy['database'][video_index_str]['annotations'],key = lambda x:x['segment(frames)'][0]) # ????????????
            all_segment_frames = [] 
            
            if cfg['process_configs']['add_noise_to_all_action']: 
                for annotations_idx in range(annotations_length):
                    all_segment_frames.append(annotations_list_sorted[annotations_idx]['segment(frames)'])

            else: 
                assert(False)
                
            feature_process[video_index_str] = set() 

            if cfg['process_configs']['noise_length'] != 0:
                for segment_idx in range(len(all_segment_frames)):
                    if cfg['process_configs']['interval_process'] == False: 
                        frame1, frame2 = all_segment_frames[segment_idx]
                        frame_position = frame1 + (frame2 - frame1) * cfg['process_configs']['noise_position_at_segment'] / 100
                        feature_middle = max((frame_position - 1 - chunk_size) // frequency + 1, 0) + 1

                        feature_start = feature_middle - cfg['process_configs']['noise_length'] / 2
                        feature_end = feature_middle + cfg['process_configs']['noise_length']  / 2 - 1

                        feature_start = int(feature_start + 0.5)
                        feature_end = int(feature_end + 0.5)
                        feature_end = min(feature_max, feature_end)
                        add_noise_list = [j for j in range(feature_start, feature_end + 1)]

                        feature_process[video_index_str].update(add_noise_list) # ???????????????????
        else:
            del data_copy['database'][video_index_str]

    with open(os.path.join(cfg['file_root']['output_file_dir'],'annotations','thumos14.json'),'w') as file: # ??????
        _ = json.dump(data_copy, file, indent=4, sort_keys=False) 
        
    return feature_process




    
    
    
    