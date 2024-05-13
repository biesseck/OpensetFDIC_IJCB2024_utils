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


# FOLDERS_NAMES_TPR_FPR = []


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]/all_detections_thresh=0.01.txt', help='')
    parser.add_argument('--gt_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/validation.csv', help='')
    parser.add_argument('--output_dir', type=str, default='', help='the dir where to save results')
    parser.add_argument('--iou', type=float, default=0.5, help='')
    parser.add_argument('--save-crops', action='store_true')
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


def load_gt_detections(csv_file):
    detections_by_image = {}
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            file_name = row['FILE']
            detection = {
                'FACE_ID': int(row['FACE_ID']),
                'SUBJECT_ID': int(row['SUBJECT_ID']),
                'FACE_X': float(row['FACE_X']),
                'FACE_Y': float(row['FACE_Y']),
                'FACE_WIDTH': float(row['FACE_WIDTH']),
                'FACE_HEIGHT': float(row['FACE_HEIGHT'])
            }
            if file_name in detections_by_image:
                detections_by_image[file_name].append(detection)
            else:
                detections_by_image[file_name] = [detection]
    return detections_by_image


def load_pred_detections(csv_file):
    detections_by_image = {}
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            file_name = row['FILE']
            detection = {
                'DETECTION_SCORE': float(row['DETECTION_SCORE']),
                'BB_X': float(row['BB_X']),
                'BB_Y': float(row['BB_Y']),
                'BB_WIDTH': float(row['BB_WIDTH']),
                'BB_HEIGHT': float(row['BB_HEIGHT']),
                'REYE_X': float(row['REYE_X']),
                'REYE_Y': float(row['REYE_Y']),
                'LEYE_X': float(row['LEYE_X']),
                'LEYE_Y': float(row['LEYE_Y']),
                'NOSE_X': float(row['NOSE_X']),
                'NOSE_Y': float(row['NOSE_Y']),
                'RMOUTH_X': float(row['RMOUTH_X']),
                'RMOUTH_Y': float(row['RMOUTH_Y']),
                'LMOUTH_X': float(row['LMOUTH_X']),
                'LMOUTH_Y': float(row['LMOUTH_Y'])
            }
            if file_name in detections_by_image:
                detections_by_image[file_name].append(detection)
            else:
                detections_by_image[file_name] = [detection]
    
    return detections_by_image


def calculate_iou(bbox_gt, bbox_pred):
    x1_gt, y1_gt, w_gt, h_gt = bbox_gt
    x1_pred, y1_pred, w_pred, h_pred = bbox_pred

    x2_gt = x1_gt + w_gt
    y2_gt = y1_gt + h_gt
    x2_pred = x1_pred + w_pred
    y2_pred = y1_pred + h_pred

    x_left = max(x1_gt, x1_pred)
    y_top = max(y1_gt, y1_pred)
    x_right = min(x2_gt, x2_pred)
    y_bottom = min(y2_gt, y2_pred)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_gt_area = w_gt * h_gt
    bbox_pred_area = w_pred * h_pred
    iou = intersection_area / float(bbox_gt_area + bbox_pred_area - intersection_area)
    return iou


def make_detections_labels(pred_detections, gt_detections, iou_threshold=0.5):
    pred_keys = list(pred_detections.keys())
    # gt_keys = list(gt_detections.keys())
    # global_counts = {}
    
    for idx_pred_key, pred_key in enumerate(pred_keys):
        img_pred_dets = pred_detections[pred_key]
        img_gt_dets = gt_detections[pred_key]
        
        img_gt_dets_to_compare = copy.deepcopy(img_gt_dets)
        tp, fp, fn = 0, 0, 0
        for idx_pred, pred in enumerate(img_pred_dets):
            pred_bbox = (pred['BB_X'], pred['BB_Y'], pred['BB_WIDTH'], pred['BB_HEIGHT'])
            found_gt = False
            for idx_gt, gt in enumerate(img_gt_dets_to_compare):
                gt_bbox = (gt['FACE_X'], gt['FACE_Y'], gt['FACE_WIDTH'], gt['FACE_HEIGHT'])
                iou = calculate_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold:
                    # tp += 1
                    found_gt = True
                    img_gt_dets_to_compare.remove(gt)
                    break

            if found_gt:
                tp += 1
                pred['TYPE'] = 'tp'
            else:
                fp += 1
                pred['TYPE'] = 'fp'
        fn = len(img_gt_dets_to_compare)
        assert (tp+fp) == len(img_pred_dets)
        print(f'    {idx_pred_key}/{len(pred_keys)} - image \'{pred_key}\' - detections: {len(img_pred_dets)} - tp: {tp}, fp: {fp}, fn: {fn}        ', end='\r')
    print('')
    return pred_detections


# based on https://github.com/AIML-IfI/uccs-facerec-challenge/blob/main/facerec/face_detection.py
def save_detections_with_labels_txt(detections, saving_path, thresh=0.0):
    num_saved_faces = 0
    with open(saving_path, "w") as f:
        header = "FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y,TYPE"
        header_list = header.split(',')
        f.write(header + "\n")
        for d, img_name in enumerate(detections):
            for i, detect in enumerate(detections[img_name]):
                if detect['DETECTION_SCORE'] >= thresh:
                    num_saved_faces += 1
                    for k, key in enumerate(header_list):
                        if k == 0:
                            f.write("%s" % img_name)
                        elif key == 'TYPE':
                            f.write(",%s" % detect[key])
                        elif key == 'DETECTION_SCORE':
                            f.write(",%3.20f" % detect[key])
                        else:
                            f.write(",%3.2f" % detect[key])
                    f.write("\n")
    return num_saved_faces


def main(args):
    output_dir = args.output_dir.rstrip('/')
    if output_dir == '':
        output_dir = os.path.dirname(args.pred_path)

    print(f'Loading pred detections \'{args.pred_path}\'')
    pred_detections = load_pred_detections(args.pred_path)
    # print('pred_detections[\'0006c02f81b58512e7a28059650b99f2.jpg\']:', pred_detections['0006c02f81b58512e7a28059650b99f2.jpg'])
    # print('len(pred_detections[\'0006c02f81b58512e7a28059650b99f2.jpg\']):', len(pred_detections['0006c02f81b58512e7a28059650b99f2.jpg']))
    # sys.exit(0)

    print(f'Loading groundtruth detections \'{args.gt_path}\'')
    gt_detections = load_gt_detections(args.gt_path)
    # print('gt_detections:', gt_detections)
    # print('gt_detections[\'00085e794b56df6e46a8a8b51cf23d71.jpg\']:', gt_detections['00085e794b56df6e46a8a8b51cf23d71.jpg'])
    # sys.exit(0)

    print(f'Making detections labels')
    pred_detections_with_label = make_detections_labels(pred_detections, gt_detections, args.iou)
    # print("pred_detections_with_label['fff9a48979ed790869b862a2f9500f31.jpg']:", pred_detections_with_label['fff9a48979ed790869b862a2f9500f31.jpg'])
    # for idx_pred, pred in enumerate(pred_detections_with_label['fff9a48979ed790869b862a2f9500f31.jpg']):
    #     print('pred:', pred)
    # sys.exit(0)

    output_file_name, output_ext = os.path.splitext(args.pred_path)
    output_file_name = output_file_name + '_WITH_LABELS_TP_FP' + output_ext
    output_file_path = os.path.join(output_dir, output_file_name)
    print(f'Saving detections with labels')
    num_saved_dets = save_detections_with_labels_txt(pred_detections_with_label, output_file_path, thresh=0.0)
    print(f'    Saved {num_saved_dets} detections')

    print('\nFinished\n')
    



if __name__ == '__main__':
    args = getArgs()
    main(args)