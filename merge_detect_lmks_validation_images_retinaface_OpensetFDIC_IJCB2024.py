import os
import sys
import time
import datetime
import numpy as np
import argparse
import ast
import cv2
import csv


HEADER_FORMAT_UCCS_V1 = 'FACE_ID,IMAGE,SUBJECT_ID,topleft_Y,topleft_X,bottomright_Y,bottomright_X,reye_Y,reye_X,leye_Y,leye_X'
HEADER_FORMAT_UCCS_V2 = 'FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y'


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_bbox_lmks_retinaface/txt', help='')
    parser.add_argument('--input_ext', type=str, default='_format_uccsV1.txt', help='')
    parser.add_argument('--output_path', type=str, default='', help='the dir the cropped faces of your dataset where to save')
    parser.add_argument('--thresh', type=float, default=0.01, help='threshold for face detection')
    parser.add_argument('--save_separated_subj_ids_file', action='store_true', help='')

    args = parser.parse_args()
    return args


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
def save_detections_txt(detections, saving_path, thresh):
    header_list = list(detections[0][0].keys())
    header_str = ','.join(header_list)
    # print('header:', header)
    # sys.exit(0)
    num_saved_faces = 0
    with open(saving_path, "w") as f:
        # header = "FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y"
        # header_list = header.split(',')
        f.write(header_str + "\n")
        for d, detects in enumerate(detections):
            for i, detect in enumerate(detects):
                if 'DETECTION_SCORE' in header_list:
                    if detect['DETECTION_SCORE'] >= thresh:
                        for k, key in enumerate(header_list):
                            if k == 0:
                                f.write("%s" % detect[key])
                            else:
                                f.write(",%3.2f" % detect[key])
                        f.write("\n")
                else:
                    for k, key in enumerate(header_list):
                        if k == 0:
                            f.write("%s" % detect[key])
                        else:
                            f.write(",%s" % detect[key])

                    f.write("\n")

                num_saved_faces += 1

    return num_saved_faces


def save_subj_ids_txt(detections, saving_path, thresh):
    header_list = list(detections[0][0].keys())
    assert 'SUBJECT_ID' in header_list, f'Error, file detections in current format do not contain the column \'SUBJECT_ID\''
    header_str = 'SUBJECT_ID'
    # print('header:', header)
    # sys.exit(0)
    num_saved_ids = 0
    with open(saving_path, "w") as f:
        # header = "FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y"
        # header_list = header.split(',')
        f.write(header_str + "\n")
        for d, detects in enumerate(detections):
            for i, detect in enumerate(detects):
                f.write("%s" % detect[header_str])
                f.write("\n")

                num_saved_ids += 1

    return num_saved_ids


def load_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header
        header_str = ','.join(header)

        if header_str == HEADER_FORMAT_UCCS_V1:
            for row in csv_reader:
                row_data = {
                    header[i]: value for i, value in enumerate(row)
                }
                data.append(row_data)

        elif header_str == HEADER_FORMAT_UCCS_V2:
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
        print(f'{i}/{len(all_txt_paths)}', end='\r')
        detections_img = load_csv_file(txt_path)
        num_det_faces += len(detections_img)
        detect_list[i] = detections_img
    print('')
    return detect_list, num_det_faces



def main(args):
    input_dir = args.input_path.rstrip('/')
    if not os.path.exists(input_dir):
        print(f'The input path doesn\'t exists: {input_dir}')
        sys.exit(0)

    output_dir = args.output_path.rstrip('/')
    if output_dir == '':
        output_dir = '/'.join(input_dir.split('/')[:-1])

    print(f'Searching detection files \'{input_dir}\'')
    all_txt_paths = get_all_files_in_path(input_dir, file_extension=args.input_ext)
    print('    Found:', len(all_txt_paths), 'files')

    print(f'Loading individual detections from \'{input_dir}\'')
    all_detections, num_det_faces = load_all_detections(all_txt_paths)
    assert len(all_txt_paths) == len(all_detections)
    print('    Loaded:', len(all_detections), 'files')
    print('    num_det_faces:', num_det_faces)
    # print('all_detections:', all_detections)
    # sys.exit(0)

    output_detections_path = os.path.join(output_dir, f"all_detections_thresh={args.thresh}_{args.input_ext.strip('_').split('.')[0]}.txt")
    print(f'Saving merged file \'{output_detections_path}\'')
    num_saved_faces = save_detections_txt(all_detections, output_detections_path, args.thresh)
    print('    num_saved_faces:', num_saved_faces)
    # sys.exit(0)

    if args.save_separated_subj_ids_file:
        output_subj_id_path = os.path.join(output_dir, f"subj_ids_all_detections_thresh={args.thresh}.txt")
        save_subj_ids_txt(all_detections, output_subj_id_path, args.thresh)

    print('Finished!\n')




if __name__ == '__main__':
    args = getArgs()
    main(args)