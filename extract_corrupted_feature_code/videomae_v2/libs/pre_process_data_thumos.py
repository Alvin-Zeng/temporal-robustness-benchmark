# -*- coding:gb2312 -*-
import json
import os
import copy
import numpy as np

def read_feature_frame_max(cfg, data):
    chunk_size = cfg['chunk_size']
    frequency = cfg['frequency']

    feature_and_frame_max_dict = {}
    if not os.path.exists('libs/tridet_feature_max.json'):
        for video_index_str in data['database']:
            if video_index_str[6] == 't':
                img_file_list = os.listdir(os.path.join(cfg['file_root']['test_img_input_dir'], video_index_str))
            elif video_index_str[6] == 'v':
                img_file_list = os.listdir(os.path.join(cfg['file_root']['val_img_input_dir'], video_index_str))
            img_file_list.sort()
            img_max = int(img_file_list[-1][4:9])
            feature_max = (img_max - 1 - chunk_size) // frequency + 1 # feature???????
            while feature_max * frequency + chunk_size + 1 > img_max:
                feature_max -= 1
            feature_and_frame_max_dict[video_index_str] = [img_max, feature_max]

        with open('libs/tridet_feature_max.json','w') as file:
            _ = json.dump(feature_and_frame_max_dict, file, indent=4, sort_keys=False) 
    else:
        with open('libs/tridet_feature_max.json','r') as f: 
            feature_and_frame_max_dict = json.load(f) 
    return feature_and_frame_max_dict

def pre_process_annotations_data(cfg):   
    chunk_size = cfg['chunk_size']
    frequency = cfg['frequency']
    with open(cfg['file_root']['annotations_input_file_root'],'r') as f: 
        data = json.load(f)

    data_copy = copy.deepcopy(data)
    feature_to_frame = {} # ÿ����ƵҪ�滻��������Χ
    frames_process = {} # ÿ����ƵҪ�����ͼƬ֡����

    feature_and_frame_max_dict = read_feature_frame_max(cfg, data)

    length_threshold = cfg['process_configs']['process_annotations_length_threshold']

    if length_threshold == 0:
        feature_to_frame['video_test_0001292.mp4'] = set()
        
    # for video in feature_and_frame_max_dict:
    #     feature_to_frame[video+'.mp4'] = set()
    #     process_feature = [i for i in range(feature_and_frame_max_dict[video][1])]
    #     process_feature = process_feature[::cfg['process_configs']['feature_step']]
    #     feature_to_frame[video+'.mp4'].update(process_feature)
    # return feature_to_frame

    # feature_file_list = [feature_name[:-4] for feature_name in os.listdir(os.path.join(cfg['file_root']['output_file_dir'], 'i3d_features'))]

    for video_index_str in data['database']: # data ����������, data_copy��������

        frame_max, feature_max = feature_and_frame_max_dict[video_index_str]
        # print(video_index_str)
        time = data_copy['database'][video_index_str].get('duration')
        annotations_list_sorted = sorted(data_copy['database'][video_index_str]['annotations'],key = lambda x:x['segment(frames)'][1]) # ������ֹ֡����
        
        # ɾ������ı�ע���ڱ�ע����һЩ������עʱ�䳬����Ƶʱ����
        for j in range(len(annotations_list_sorted)):
            if annotations_list_sorted[j]['segment'][1] > time:
                del annotations_list_sorted[j:]
                break
            
        data_copy['database'][video_index_str]['annotations'] = annotations_list_sorted
        annotations_length = len(annotations_list_sorted) # �õ���Ƶ�µ�segment����  
        if annotations_length >= length_threshold:   
            annotations_list_sorted = sorted(data_copy['database'][video_index_str]['annotations'],key = lambda x:x['segment(frames)'][0]) # ������ʼ֡����
            all_segment_frames = [] # һ����Ƶ�Ķ���ʱ��Ƭ����һ���б��ʾ
            
            if cfg['process_configs']['add_noise_to_all_action']: # �����Ӷ�����ܶ�����б�
                for annotations_idx in range(annotations_length):
                    all_segment_frames.append(annotations_list_sorted[annotations_idx]['segment(frames)'])

            else: # һ��������һ�����б���Ҳ��Ƕ���б�
                segment_index = int(cfg['process_configs']['noise_position_at_video'] / 100 * (annotations_length-1) + 0.5) # ���ȶ�Ӧ������
                
                if length_threshold>=4:
                    # ��β��ʱ�����⴦��
                    if segment_index <= 0: # ��һ����ʱ���ɵڶ��������ܼ�������ָ��
                        segment_index = 1
                    elif segment_index >= annotations_length - 1: # ���һ����ʱ���ɵ����ڶ���
                        segment_index = annotations_length - 2
                all_segment_frames.append(annotations_list_sorted[segment_index]['segment(frames)']) # ע��������Ƕ���б�Ϊ�˸�ѡ�����ж�����ʱ��һ��
                
            feature_to_frame[video_index_str+'.mp4'] = set()   # �����Ƶ����Ҫ������ȡ����������
            frames_process[video_index_str+'.mp4'] = set()    # һ����Ŵ���֡�������ļ���
            if cfg['process_configs']['noise_length'] != 0:
                for segment_idx in range(len(all_segment_frames)): # ����һ����Ƶ�б��е��������б�
                    frame_cur_start, frame_cur_end = all_segment_frames[segment_idx]

                    if cfg['process_configs']['sample_way'] == 'continuous_in_action':
                        frame_position = frame_cur_start + (frame_cur_end - frame_cur_start) / 2
                        if cfg['process_configs']['add_noise_unity'] == 'frame':
                            frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                            frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                        elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                            frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame_cur_end - frame_cur_start) / 200
                            frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame_cur_end - frame_cur_start) / 200
                        # Ҫ�����֡�ķ�Χ
                        frame_start = int(frame_start + 0.5)
                        frame_end = int(frame_end + 0.5)
                        add_noise_list = [j for j in range(frame_start, frame_end + 1)]
                        frames_process[video_index_str+'.mp4'].update(add_noise_list)

                    elif cfg['process_configs']['sample_way'] == 'continuous_in_background':
                        frame_position = None
                        add_noise_list = []
                        if segment_idx != 0:
                            _, frame_last_end = all_segment_frames[segment_idx-1]
                            if frame_last_end + 1 <= frame_cur_start - 1:
                                frame_position = (frame_last_end + frame_cur_start) / 2
                        elif segment_idx == 0:
                            if frame_cur_start > 1:
                                frame_position = frame_cur_start / 2

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
                            frames_process[video_index_str+'.mp4'].update(add_noise_list)

                        if segment_idx == len(all_segment_frames) - 1:
                            frame_position = (frame_cur_end + 1 + frame_max) / 2
                            if cfg['process_configs']['add_noise_unity'] == 'frame':
                                frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                                frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                            elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                                frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame_cur_end - frame_cur_start) / 200
                                frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame_cur_end - frame_cur_start) / 200
                            frame_start = int(frame_start + 0.5)
                            frame_end = min(int(frame_end + 0.5), frame_max)
                            add_noise_list = [j for j in range(frame_start, frame_end + 1)]
                            frames_process[video_index_str+'.mp4'].update(add_noise_list)

                    elif cfg['process_configs']['sample_way'] == 'random_in_action':
                        add_noise_list = np.random.randint(frame_cur_start, frame_cur_end, cfg['process_configs']['noise_length'])
                        frames_process[video_index_str+'.mp4'].update(add_noise_list)

                    elif cfg['process_configs']['sample_way'] == 'random_in_background':
                        if segment_idx != 0:
                            _, frame_last_end = all_segment_frames[segment_idx-1]
                            if frame_last_end + 1 <= frame_cur_start - 1:
                                add_noise_list = np.random.randint(frame_last_end+1, frame_cur_start-1, cfg['process_configs']['noise_length'])

                        elif segment_idx == 0:
                            if frame_cur_start > 1:
                                add_noise_list = np.random.randint(1, frame_cur_start-1, cfg['process_configs']['noise_length'])

                        frames_process[video_index_str+'.mp4'].update(add_noise_list)

                        if segment_idx == len(all_segment_frames) - 1:
                            add_noise_list = np.random.randint(frame_cur_end, frame_max, cfg['process_configs']['noise_length'])
                            frames_process[video_index_str+'.mp4'].update(add_noise_list) # ��Ҫ���������֡�ļ���
                    elif cfg['process_configs']['sample_way'] == 'random_in_action_and_background':
                        if segment_idx != 0:
                            _, frame_last_end = all_segment_frames[segment_idx-1]
                            if frame_last_end + 1 <= frame_cur_end:
                                add_noise_list = np.random.randint(frame_last_end + 1, frame_cur_end, cfg['process_configs']['noise_length'])

                        elif segment_idx == 0:
                            if frame_cur_end > 1:
                                add_noise_list = np.random.randint(1, frame_cur_end, cfg['process_configs']['noise_length'])

                        frames_process[video_index_str+'.mp4'].update(add_noise_list)
                    elif cfg['process_configs']['sample_way'] == 'continuous_in_action_and_background':
                        add_noise_list = []
                        frame_position = None
                        if segment_idx != 0:
                            _, frame_last_end = all_segment_frames[segment_idx-1]
                            if frame_last_end + 1 <= frame_cur_end:
                                select_num = np.random.randint(frame_last_end + 1, frame_cur_end, 1)
                                if select_num[0] < frame_cur_start:
                                    frame_position = (frame_cur_start + frame_last_end) / 2
                                else:
                                    frame_position = (frame_cur_end + frame_cur_start) / 2
                        elif segment_idx == 0:
                            if frame_cur_start > 1:
                                select_num = np.random.randint(1, frame_cur_end, 1)
                                if select_num[0] < frame_cur_start:
                                    frame_position = frame_cur_start / 2
                                else:
                                    frame_position = (frame_cur_end + frame_cur_start) / 2
                        if frame_position != None:
                            if cfg['process_configs']['add_noise_unity'] == 'frame':
                                frame_start = frame_position - cfg['process_configs']['noise_length'] / 2
                                frame_end = frame_position + cfg['process_configs']['noise_length']  / 2 - 1
                            elif cfg['process_configs']['add_noise_unity'] == 'percentage': 
                                frame_start = frame_position - cfg['process_configs']['noise_length'] * (frame_cur_end - frame_cur_start) / 200
                                frame_end = frame_position + cfg['process_configs']['noise_length']  * (frame_cur_end - frame_cur_start) / 200
                            frame_start = int(frame_start + 0.5)
                            frame_end = min(int(frame_end + 0.5), frame_max)
                            add_noise_list = [j for j in range(frame_start, frame_end + 1)]
                            frames_process[video_index_str+'.mp4'].update(add_noise_list)
                    
            # ������Ҫ�滻������λ��,�±��0��ʼ����
            for sample_frame in frames_process[video_index_str+'.mp4']:
                if cfg['process_configs']['feature_process_rate'] == 4:
                    feature_start = max((sample_frame - 1 - chunk_size) // frequency + 1, 0)
                    feature_end = min(max((sample_frame - 1) // frequency, 0), feature_max)
                    reextract_feature_idx = [j for j in range(feature_start, feature_end + 1)]
                    feature_to_frame[video_index_str+'.mp4'].update(reextract_feature_idx)
                elif cfg['process_configs']['feature_process_rate'] == 1:
                    feature_start = max((sample_frame - 1 - chunk_size) // frequency + 1, 0)
                    feature_end = min(max((sample_frame - 1) // frequency, 0), feature_max)
                    reextract_feature_idx = (feature_start + feature_end) // 2
                    feature_to_frame[video_index_str+'.mp4'].add(reextract_feature_idx)           

        else:
            del data_copy['database'][video_index_str]

    with open(os.path.join(cfg['file_root']['output_file_dir'],'annotations','thumos14.json'),'w') as file: # ��·��
        _ = json.dump(data_copy, file, indent=4, sort_keys=False) 

    return frames_process, feature_to_frame