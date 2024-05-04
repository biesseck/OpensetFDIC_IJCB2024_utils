import os
import sys
import time
import datetime
import numpy as np
import argparse
import ast
import cv2
import csv
import copy
import glob
import pickle


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dets', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]/txt', help='')
    parser.add_argument('--ext_dets', type=str, default='.txt', help='')
    parser.add_argument('--input_crops', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]/imgs', help='')
    parser.add_argument('--ext_crops', type=str, default='.png', help='')
    parser.add_argument('--output_path', type=str, default='', help='the dir the cropped faces of your dataset where to save')

    args = parser.parse_args()
    return args


def save_object_with_pickle(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_object_with_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def get_all_files_in_path(folder_path, file_extension='.jpg', pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            if pattern in path_file and path_file.endswith(file_extension):
                file_list.append(path_file)
    file_list.sort()
    return file_list


# based on https://github.com/AIML-IfI/uccs-facerec-challenge/blob/main/facerec/face_detection.py
def save_detections_txt(detections, saving_path):
    with open(saving_path, "w") as f:
        header = "FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y"
        header_list = header.split(',')
        f.write(header + "\n")
        for i, detect in enumerate(detections):
            for k, key in enumerate(header_list):
                if k == 0:
                    f.write("%s" % detect[key])
                elif key == 'DETECTION_SCORE':
                    f.write(",%3.20f" % detect[key])
                else:
                    f.write(",%3.2f" % detect[key])
            f.write("\n")


def save_corrected_detections_txt_per_image(all_detections_scores_corrected, output_dir):
    for idx_dets_img, dets_img in enumerate(all_detections_scores_corrected):
        # print('type(dets_img):', type(dets_img))
        img_name = dets_img[0]['FILE']
        dets_file_name = img_name.split('.')[0] + '.txt'
        dets_file_path = os.path.join(output_dir, dets_file_name)
        print(f'    {idx_dets_img}/{len(all_detections_scores_corrected)} - Saving corrected detections \'{dets_file_path}\'')
        save_detections_txt(dets_img, dets_file_path)
        # sys.exit(0)


def load_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header
        for row in csv_reader:
            row_data = {
                header[i]: float(value) if i > 0 else value  # Convert values to float except for the first column
                for i, value in enumerate(row)
            }
            data.append(row_data)
    return data


def load_all_detections(all_txt_paths=[]):
    detect_list = [None] * len(all_txt_paths)
    num_det_faces = 0
    for i, txt_path in enumerate(all_txt_paths):
        print(f'    File {i}/{len(all_txt_paths)}', end='\r')
        detections_img = load_csv_file(txt_path)
        num_det_faces += len(detections_img)
        detect_list[i] = detections_img
        # print('detections_img:', detections_img)
        # sys.exit(0)
    print('')
    return detect_list, num_det_faces


def organize_crops_paths(crops_paths):
    crops_scores_dict = {}
    for idx_crop_path, crop_path in enumerate(crops_paths):
        print(f'    crop {idx_crop_path}/{len(crops_paths)} ({((idx_crop_path/len(crops_paths))*100):.{2}f}%)', end='\r')
        file_name = os.path.basename(crop_path)                                   # '002f6b6ec899c02e21917a97d9b60ca8_bbox2209_conf0.0010001278715208173.png'
        file_name_parts = file_name.split('_')                                    # ['002f6b6ec899c02e21917a97d9b60ca8', 'bbox2209', 'conf0.0010001278715208173.png']
        img_name = file_name_parts[0]                                             # '002f6b6ec899c02e21917a97d9b60ca8'
        bbox_idx = int(file_name_parts[1].split('bbox')[-1])                      # 2209
        score_det = float(file_name_parts[2].split('conf')[-1].split('.png')[0])  # 0.0010001278715208173
        crop_key = img_name + f'_bbox{bbox_idx}'                                  # '002f6b6ec899c02e21917a97d9b60ca8_bbox2209'
        crops_scores_dict[crop_key] = score_det                                   # {'002f6b6ec899c02e21917a97d9b60ca8_bbox2209': 0.0010001278715208173}
    print('')
    return crops_scores_dict


def replace_detection_scores_from_crops_paths(all_detections, crops_dir, ext_crops):
    print(f'    Searching crops in \'{crops_dir}\'...')
    crops_paths = get_all_files_in_path(crops_dir, file_extension=ext_crops)
    print(f'    Found {len(crops_paths)} crops')
    # sys.exit(0)

    print(f'    Organizing crops paths and scores...')
    crops_scores_dict = organize_crops_paths(crops_paths)
    print(f'    Organized {len(crops_scores_dict)} crops')
    # sys.exit(0)

    print(f'    Replacing score detections...')
    for idx_dets_img, dets_img in enumerate(all_detections):
        for idx_bbox_det, one_det in enumerate(dets_img):
            # print(f'one_det:', one_det)
            img_name = one_det['FILE']
            crop_key = img_name.split('.')[0]+f'_bbox{idx_bbox_det}'
            try:
                score_float = crops_scores_dict[crop_key]
                print(f"    Replacing score detection - img {idx_dets_img}/{len(all_detections)}    detection {idx_bbox_det}/{len(dets_img)}    current: {one_det['DETECTION_SCORE']}    correct: {score_float}")
                one_det['DETECTION_SCORE'] = score_float
            except KeyError as kerr:
                print('    KeyError:', kerr)

        print('------------------')
        # sys.exit(0)
    return all_detections
    


def main(args):
    input_dets = args.input_dets.rstrip('/')
    if not os.path.exists(input_dets):
        print(f'The detections path doesn\'t exists: {input_dets}')
        sys.exit(0)

    input_crops = args.input_crops.rstrip('/')
    if not os.path.exists(input_crops):
        print(f'The crops path doesn\'t exists: {input_crops}')
        sys.exit(0)

    output_dir = args.output_path.rstrip('/')
    if output_dir == '':
        output_dir_name = input_dets.split('/')[-1] + '_corrected_detection_scores'
        output_dir = os.path.join(os.path.dirname(input_dets), output_dir_name)
    print(f'Creating output dir: \'{output_dir}\'')
    os.makedirs(output_dir, exist_ok=True)
    # sys.exit(0)

    all_detections_scores_corrected_file = 'all_detections_scores_corrected.pkl'
    all_detections_scores_corrected_path = os.path.join(os.path.dirname(input_dets), all_detections_scores_corrected_file)
    all_detections_scores_corrected = None
    # print('all_detections_scores_corrected_path:', all_detections_scores_corrected_path)
    # sys.exit(0)
    if not os.path.isfile(all_detections_scores_corrected_path):
        print(f'Searching detection files \'{input_dets}\'')
        all_txt_paths = get_all_files_in_path(input_dets, file_extension=args.ext_dets)
        # print('all_txt_paths:', all_txt_paths)
        print('    Found:', len(all_txt_paths), 'files')
        # sys.exit(0)

        print(f'Loading individual detections from \'{input_dets}\'')
        all_detections, num_det_faces = load_all_detections(all_txt_paths)
        assert len(all_txt_paths) == len(all_detections)
        print('    Loaded:', len(all_detections), 'files')
        print('    num_det_faces:', num_det_faces)
        # sys.exit(0)

        print(f'Loading crops from \'{input_crops}\'')
        all_detections_scores_corrected = replace_detection_scores_from_crops_paths(all_detections, input_crops, args.ext_crops)

        print(f'Saving all_detections_scores_corrected: \'{all_detections_scores_corrected_path}\'')
        save_object_with_pickle(all_detections_scores_corrected, all_detections_scores_corrected_path)
        print(f'    Saved')
    else:
        print(f'Loading all_detections_scores_corrected: \'{all_detections_scores_corrected_path}\'')
        all_detections_scores_corrected = load_object_with_pickle(all_detections_scores_corrected_path)
        print(f'    Loaded')
    
    # sys.exit(0)

    # output_detections_path = os.path.join(output_dir, f'all_detections_thresh={args.thresh}.txt')
    print(f'Saving corrected detections to \'{output_dir}\'')
    save_corrected_detections_txt_per_image(all_detections_scores_corrected, output_dir)
    
    print('\nFinished!\n')




if __name__ == '__main__':
    args = getArgs()
    main(args)