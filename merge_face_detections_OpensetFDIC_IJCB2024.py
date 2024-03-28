import os
import sys
import time
import datetime
import numpy as np
import argparse
import ast
import cv2
import csv


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.5]/txt', help='')
    parser.add_argument('--input_ext', type=str, default='.txt', help='')
    parser.add_argument('--output_path', type=str, default='', help='the dir the cropped faces of your dataset where to save')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold for face detection')

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
    num_saved_faces = 0
    with open(saving_path, "w") as f:
        header = "FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y"
        header_list = header.split(',')
        f.write(header + "\n")
        for d, detects in enumerate(detections):
            for i, detect in enumerate(detects):
                if detect['DETECTION_SCORE'] >= thresh:
                    num_saved_faces += 1
                    for k, key in enumerate(header_list):
                        if k == 0:
                            f.write("%s" % detect[key])
                        else:
                            f.write(",%3.2f" % detect[key])
                    f.write("\n")
    return num_saved_faces


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
        detections_img = load_csv_file(txt_path)
        num_det_faces += len(detections_img)
        detect_list[i] = detections_img
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

    output_detections_path = os.path.join(output_dir, f'all_detections_thresh={args.thresh}.txt')
    print(f'Saving merged file \'{output_detections_path}\'')
    num_saved_faces = save_detections_txt(all_detections, output_detections_path, args.thresh)
    print('    num_saved_faces:', num_saved_faces)

    print('Finished!\n')




if __name__ == '__main__':
    args = getArgs()
    main(args)