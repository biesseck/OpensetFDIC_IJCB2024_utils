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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arcface import Arcface
from retinaface.retinaface import RetinaFace
from insightface.utils import face_align


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/gallery_images', help='')
    parser.add_argument('--img_ext', type=str, default='.png', help='')
    parser.add_argument('--detections', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/gallery.csv', help='')
    parser.add_argument('--output_dir', type=str, default='', help='the dir where to save results')
    parser.add_argument('--save_crops', action='store_true')
    parser.add_argument('--rotate90', action='store_true')

    args = parser.parse_args()
    return args


def load_fr_model():
    r100_arcface_path = 'models/models_weights/arcface-torch/ms1mv3_arcface_r100_fp16/backbone.pth'
    device = 'cuda:0'
    print(f'Loading face recognition model from \'{r100_arcface_path}\'')
    arcface = Arcface(pretrained_path=r100_arcface_path).to(device)
    arcface.eval()
    return arcface


def load_detections(csv_file):
    detect_set = None
    data = []
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Get the header row
            if headers[0] == 'FILE' and headers[1] == 'SUBJECT_ID':
                detect_set = 'gallery_ground_truth'
            elif headers[0] == 'FILE' and headers[1] == 'FACE_ID':
                detect_set = 'validation_ground_truth'
            elif headers[0] == 'FILE' and headers[1] == 'DETECTION_SCORE':
                detect_set = 'validation_detections'

            for row in reader:
                row_dict = {}
                for i, value in enumerate(row):
                    if headers[i] == 'FILE' or headers[i] == 'SUBJECT_ID':
                        row_dict[headers[i]] = value
                    else:
                        row_dict[headers[i]] = float(value)
                data.append(row_dict)
    except Exception as e:
        print("Error loading CSV file:", e)
    return detect_set, data


def load_gallery_imgs(detections, img_dir):
    imgs_bgr = [None] * len(detections)
    for idx_detection, detection in enumerate(detections):
        img_path = os.path.join(img_dir, detection['SUBJECT_ID'], detection['FILE'])
        assert os.path.isfile(img_path), f'Error, file not found \'{img_path}\''
        print(f'{idx_detection}/{len(detections)} - Loading \'{img_path}\'', end='\r')
        imgs_bgr[idx_detection] = cv2.imread(img_path)
    print('')
    return imgs_bgr


def prepare_landmarks(lmks):
    lmks_ = np.array(lmks).reshape((5, 2))
    return lmks_


def prepare_face(face_img, lmks, rotate90=False):
    aimg = face_align.norm_crop(face_img, landmark=lmks)
    if rotate90:
        aimg = cv2.rotate(aimg, cv2.ROTATE_90_CLOCKWISE)

    input_mean, input_std = 127.5, 127.5
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    blob = torch.from_numpy(blob).cuda()
    return aimg, blob


def compute_gallery_ground_truth_face_embeddings(arcface, detections, img_dir, gallery_output_path, args):
    for idx_detection, detection in enumerate(detections):
        img_path = os.path.join(img_dir, detection['SUBJECT_ID'], detection['FILE'])
        print(f'{idx_detection}/{len(detections)} - Computing embedding \'{img_path}\'', end='\r')
        img = cv2.imread(img_path)

        # FILE,SUBJECT_ID,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y
        points = [[detection['REYE_X'], detection['REYE_Y']],
                  [detection['LEYE_X'], detection['LEYE_Y']],
                  [detection['NOSE_X'], detection['NOSE_Y']],
                  [detection['RMOUTH_X'], detection['RMOUTH_Y']],
                  [detection['LMOUTH_X'], detection['LMOUTH_Y']]]
        
        points_ = prepare_landmarks(points)
        aimg, blob = prepare_face(img, points_, args.rotate90)

        subj_dir_path = os.path.join(gallery_output_path, detection['SUBJECT_ID'])
        os.makedirs(subj_dir_path, exist_ok=True)

        if args.save_crops:
            # orig_img_path = os.path.join(subj_dir_path, detection['FILE'])
            align_crop_path = os.path.join(subj_dir_path, detection['FILE'].split('.')[0]+'_align.'+detection['FILE'].split('.')[1])
            cv2.imwrite(align_crop_path, aimg)
            # sys.exit(0)

        embedd = arcface(blob)
        embedd_file_name = detection['FILE'].split('.')[0] + '_embedd_2D_arcface.npy'
        embedd_path = os.path.join(subj_dir_path, embedd_file_name)
        np.save(embedd_path, embedd.cpu().detach().numpy())
    print('')


def compute_validation_detections_truth_face_embeddings(arcface, detections, img_dir, output_path, args):
    last_img_name = None
    for idx_detection, detection in enumerate(detections):
        img_path = os.path.join(img_dir, detection['FILE'])
        print(f'{idx_detection}/{len(detections)} - Computing embedding \'{img_path}\'', end='\r')
        
        if detection['FILE'] != last_img_name:
            img = cv2.imread(img_path)
        last_img_name = detection['FILE']

        # FILE,SUBJECT_ID,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y
        points = [[detection['REYE_X'], detection['REYE_Y']],
                  [detection['LEYE_X'], detection['LEYE_Y']],
                  [detection['NOSE_X'], detection['NOSE_Y']],
                  [detection['RMOUTH_X'], detection['RMOUTH_Y']],
                  [detection['LMOUTH_X'], detection['LMOUTH_Y']]]
        
        points_ = prepare_landmarks(points)
        aimg, blob = prepare_face(img, points_, args.rotate90)

        subj_dir_path = os.path.join(output_path, detection['FILE'].split('.')[0])
        os.makedirs(subj_dir_path, exist_ok=True)

        if args.save_crops:
            # orig_img_path = os.path.join(subj_dir_path, detection['FILE'])
            align_crop_path = os.path.join(subj_dir_path, detection['FILE'].split('.')[0] + '_' + str(idx_detection) + '_conf' + str(detection['DETECTION_SCORE']) + '_align.'+detection['FILE'].split('.')[1])
            cv2.imwrite(align_crop_path, aimg)
            # sys.exit(0)

        embedd = arcface(blob)
        embedd_file_name = detection['FILE'].split('.')[0] + '_' + str(idx_detection) + '_conf' + str(detection['DETECTION_SCORE']) + '_embedd_2D_arcface.npy'
        embedd_path = os.path.join(subj_dir_path, embedd_file_name)
        np.save(embedd_path, embedd.cpu().detach().numpy())
    print('')



def main(args):
    img_dir = args.img_dir.rstrip('/')
    if not os.path.exists(img_dir):
        print(f'The image path doesn\'t exists: {img_dir}')
        sys.exit(0)
    
    detections_path = args.detections
    if not os.path.exists(detections_path):
        print(f'The detections file doesn\'t exists: {detections_path}')
        sys.exit(0)

    arcface = load_fr_model()

    print(f'\nLoading detections \'{args.detections}\'')
    detect_set, detections = load_detections(args.detections)
    print(f'Loaded {len(detections)} faces')
    # for idx_detection, detection in enumerate(detections):
    #     print('detection:', detection)

    output_dir = '2D_embeddings'
    output_path = os.path.join(os.path.dirname(img_dir), output_dir)
    # embedd_output_path = os.path.join(output_path, output_path, os.path.basename(img_dir))
    embedd_output_path = os.path.join(output_path, output_path, detect_set)
    if args.rotate90: embedd_output_path += '_rotate90'
    print(f'Making output dir \'{embedd_output_path}\'')
    os.makedirs(embedd_output_path, exist_ok=True)

    print(f'\nComputing face embeddings...')
    if detect_set == 'gallery_ground_truth':
        compute_gallery_ground_truth_face_embeddings(arcface, detections, img_dir, embedd_output_path, args)
    elif detect_set == 'validation_ground_truth':
        pass
    elif detect_set == 'validation_detections':
        compute_validation_detections_truth_face_embeddings(arcface, detections, img_dir, embedd_output_path, args)

    
    # print(f'Loaded {len(gallery_imgs)} imgs')

    print('\nFinished!')


if __name__ == '__main__':
    args = getArgs()
    main(args)