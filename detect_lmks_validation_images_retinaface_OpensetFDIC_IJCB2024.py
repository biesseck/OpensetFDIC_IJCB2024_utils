# Dataset BUPT-BalancedFace
# python align_crop_faces_retinaface.py --input_path /datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000 --output_path /datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_112x112 --thresh 0.8 --scales [0.5]

# Dataset FFHQ
# python align_crop_faces_retinaface.py --input_path /datasets2/frcsyn_wacv2024/datasets/real/2_FFHQ/images1024x1024 --output_path /datasets2/frcsyn_wacv2024/datasets/real/2_FFHQ/images_crops_112x112 --thresh 0.8 --scales [0.5]

# Dataset AgeDB
# python align_crop_faces_retinaface.py --input_path /datasets2/frcsyn_wacv2024/datasets/real/4_AgeDB/03_Protocol_Images --output_path /datasets2/frcsyn_wacv2024/datasets/real/4_AgeDB/03_Protocol_Images_crops_112x112 --thresh 0.5 --scales [1.0]

import os
import sys
import time
import datetime
import numpy as np
import mxnet as mx
from tqdm import tqdm
import argparse
import ast
import cv2
from retinaface.retinaface import RetinaFace
from insightface.utils import face_align
import csv


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images', help='the dir your dataset of face which need to crop')
    parser.add_argument('--input_ext', type=str, default='.jpg', help='')
    parser.add_argument('--gt_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/validation.csv', help='')
    parser.add_argument('--output_path', type=str, default='', help='the dir the cropped faces of your dataset where to save')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument('--face_size', type=int, default=112, help='the size of the face to save, the size x%2==0, and width equal height')
    parser.add_argument('--thresh', type=float, default=0.01, help='threshold for face detection')
    parser.add_argument('--scales', type=str, default='[1.0]', help='the scale to resize image before detecting face')
    parser.add_argument('--align_face', action='store_true', help='')
    parser.add_argument('--draw_bbox_lmk', action='store_true', help='')
    parser.add_argument('--force_lmk', action='store_true', help='')
    parser.add_argument('--coords_with_respect_to_crop', action='store_true', help='')

    parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--div', default=1, type=int, help='Number of parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    args = parser.parse_args()

    args.scales = ast.literal_eval(args.scales)
    return args


def get_parts_indices(sub_folders, divisions):
    begin_div = []
    end_div = []
    div_size = int(len(sub_folders) / divisions)
    remainder = int(len(sub_folders) % divisions)

    for i in range(0, divisions):
        begin_div.append(i*div_size)
        end_div.append(i*div_size + div_size)
    
    end_div[-1] += remainder
    return begin_div, end_div


def draw_bbox(img, bbox, color=(0,255,0), thickness=6):
    result_img = img.copy()
    # x1, y1, x2, y2 = bbox
    x1, y1, w, h = bbox
    # color = (0, 255, 0)  # Green color
    # thickness = 6
    cv2.rectangle(result_img, (int(round(x1)), int(round(y1))),
                              (int(round(x1+w)), int(round(y1+h))), color, thickness)
    return result_img


def draw_lmks(img, lmks):
    if not (type(lmks[0]) is list or type(lmks[0]) is np.ndarray):
        lmks = [lmks]
    result_img = img.copy()
    for l in range(len(lmks)):
        color = (0, 0, 255)
        if l == 0 or l == 3:
            color = (0, 255, 0)
        cv2.circle(result_img, (int(round(lmks[l][0])), int(round(lmks[l][1]))), 1, color, 2)
    return result_img


def get_all_files_in_path(folder_path, file_extension='.jpg', pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            if pattern in path_file and path_file.endswith(file_extension):
                file_list.append(path_file)
    file_list.sort()
    return file_list


def get_all_paths_from_file(file_path, pattern=''):
    with open(file_path, 'r') as file:
        all_lines = [line.strip() for line in file.readlines()]
        valid_lines = []
        for i, line in enumerate(all_lines):
            if pattern in line:
                valid_lines.append(line)
        valid_lines.sort()
        return valid_lines


def add_string_end_file(file_path, string_to_add):
    string_to_add += '\n'
    try:
        with open(file_path, 'a') as file:
            file.write(string_to_add)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            file.write(string_to_add)


def get_generic_bbox_lmk(img):
    confidence = 0.
    bbox = np.array([[0., 0., img.shape[1]-1, img.shape[0]-1, confidence]])

    landmarks_percent_face_not_det = np.array([[[0.341916071428571, 0.461574107142857],
                                                [0.656533928571429, 0.459833928571429],
                                                [0.500225,          0.640505357142857],
                                                [0.370975892857143, 0.824691964285714],
                                                [0.631516964285714, 0.823250892857143]]], dtype=np.float32)
    landmarks_coords_face_not_det = np.zeros((landmarks_percent_face_not_det.shape), dtype=int)
    landmarks_coords_face_not_det[:,:,0] = landmarks_percent_face_not_det[:,:,0] * img.shape[1]
    landmarks_coords_face_not_det[:,:,1] = landmarks_percent_face_not_det[:,:,1] * img.shape[0]

    return bbox, landmarks_coords_face_not_det


def get_generic_lmk(bbox):
    x1, y1, w, h = bbox
    landmarks_percent_face_not_det = np.array([[[0.341916071428571, 0.461574107142857],
                                                [0.656533928571429, 0.459833928571429],
                                                [0.500225,          0.640505357142857],
                                                [0.370975892857143, 0.824691964285714],
                                                [0.631516964285714, 0.823250892857143]]], dtype=np.float32)
    landmarks_coords_face_not_det = np.zeros((landmarks_percent_face_not_det.shape), dtype=int)
    landmarks_coords_face_not_det[:,:,0] = (landmarks_percent_face_not_det[:,:,0] * w) + x1
    landmarks_coords_face_not_det[:,:,1] = (landmarks_percent_face_not_det[:,:,1] * h) + y1

    return landmarks_coords_face_not_det



def crop_resize_face(face_img, bbox, face_size=None):
    bbox = [round(value) for value in bbox]
    face = face_img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    if not face_size is None:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        center = (face.shape[1]/2.0, face.shape[0]/2.0)
        center_x, center_y = center
        # face = draw_lmks(face, center)
        # print('bbox_w:', bbox_w, '    bbox_h:', bbox_h)
        # print('center:', center)
        crop_size = min(w, h)
        if center_x - crop_size // 2 < 0:
            crop_size = min(crop_size, center_x * 2)
        if center_y - crop_size // 2 < 0:
            crop_size = min(crop_size, center_y * 2)
        if center_x + crop_size // 2 > face.shape[1]:
            crop_size = min(crop_size, (face.shape[1] - center_x) * 2)
        if center_y + crop_size // 2 > face.shape[0]:
            crop_size = min(crop_size, (face.shape[0] - center_y) * 2)
        
        crop_x1 = int(max(0, center_x - crop_size // 2))
        crop_y1 = int(max(0, center_y - crop_size // 2))
        crop_x2 = int(min(face.shape[1], crop_x1 + crop_size))
        crop_y2 = int(min(face.shape[0], crop_y1 + crop_size))

        cropped_image = face[crop_y1:crop_y2, crop_x1:crop_x2]

        face = cv2.resize(cropped_image, (face_size, face_size))
    # sys.exit(0)
    return face


# based on https://github.com/AIML-IfI/uccs-facerec-challenge/blob/main/facerec/face_detection.py
def save_detections_txt_format_uccsV1(detections, saving_path):
    with open(saving_path, "w") as f:
        # f.write("FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y\n")
        f.write("FACE_ID,IMAGE,SUBJECT_ID,topleft_Y,topleft_X,bottomright_Y,bottomright_X,reye_Y,reye_X,leye_Y,leye_X\n")
        for image in sorted(detections.keys()):
            # for (score,bbox,lmark) in detections[image]:
            for (gt_dets,lmarks) in detections[image]:
                for bbox_idx in range(len(gt_dets)):
                    gt_det = gt_dets[bbox_idx]
                    lmark = lmarks[bbox_idx]
                    f.write("%d,%s,%d,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n" % (gt_det['FACE_ID'], gt_det['IMAGE'], gt_det['SUBJECT_ID'],
                                                                                            gt_det['FACE_Y'], gt_det['FACE_X'],
                                                                                            gt_det['FACE_Y']+gt_det['FACE_HEIGHT'], gt_det['FACE_X']+gt_det['FACE_WIDTH'],
                                                                                            lmark[1][1], lmark[1][0],   # right eye
                                                                                            lmark[0][1], lmark[0][0]))  # left eye


# based on https://github.com/AIML-IfI/uccs-facerec-challenge/blob/main/facerec/face_detection.py
def save_detections_txt_format_uccsV2(detections, saving_path):
    with open(saving_path, "w") as f:
        f.write("FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y\n")
        for image in sorted(detections.keys()):
            # for (score,bbox,lmark) in detections[image]:
            for (scores,bboxes,lmarks) in detections[image]:
                for bbox_idx in range(len(bboxes)):
                    score = scores[bbox_idx]
                    bbox = bboxes[bbox_idx]
                    lmark = lmarks[bbox_idx]
                    f.write("%s,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n" % (image, score,
                                                                                        bbox[0],bbox[1],bbox[2],bbox[3],
                                                                                        lmark[1][0],lmark[1][1],  # right eye
                                                                                        lmark[0][0],lmark[0][1],  # left eye
                                                                                        lmark[2][0],lmark[2][1],  # nose
                                                                                        lmark[4][0],lmark[4][1],  # right mouth
                                                                                        lmark[3][0],lmark[3][1])) # left mouth


def load_gt_detections(csv_file):
    detections_by_image = {}
    num_total_detections = 0
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
            num_total_detections += 1
    return detections_by_image, num_total_detections


def crop_gt_bbox(img_bgr, gt_det, percent_expand=0.25):
    x1, y1, w, h = gt_det['FACE_X'], gt_det['FACE_Y'], gt_det['FACE_WIDTH'], gt_det['FACE_HEIGHT']
    x2, y2 = x1+w, y1+h
    w_expand = w*percent_expand
    h_expand = h*percent_expand

    x1 = round(max(x1-w_expand, 0))
    y1 = round(max(y1-h_expand, 0))
    x2 = round(min(x2+w_expand, img_bgr.shape[1]))
    y2 = round(min(y2+h_expand, img_bgr.shape[0]))
    img_bgr_crop = img_bgr[y1:y2, x1:x2]

    new_gt_det_wrt_crop = {}
    new_gt_det_wrt_crop['FACE_ID'] = gt_det['FACE_ID']
    new_gt_det_wrt_crop['SUBJECT_ID'] = gt_det['SUBJECT_ID']
    new_gt_det_wrt_crop['FACE_X'] = gt_det['FACE_X'] - x1
    new_gt_det_wrt_crop['FACE_Y'] = gt_det['FACE_Y'] - y1
    new_gt_det_wrt_crop['FACE_WIDTH'] = gt_det['FACE_WIDTH']
    new_gt_det_wrt_crop['FACE_HEIGHT'] = gt_det['FACE_HEIGHT']

    new_gt_det_wrt_whole_img = {}
    new_gt_det_wrt_whole_img['FACE_ID'] = gt_det['FACE_ID']
    new_gt_det_wrt_whole_img['SUBJECT_ID'] = gt_det['SUBJECT_ID']
    new_gt_det_wrt_whole_img['FACE_X'] = y1
    new_gt_det_wrt_whole_img['FACE_Y'] = x1
    new_gt_det_wrt_whole_img['FACE_WIDTH'] = gt_det['FACE_WIDTH']
    new_gt_det_wrt_whole_img['FACE_HEIGHT'] = gt_det['FACE_HEIGHT']

    return img_bgr_crop, new_gt_det_wrt_crop, new_gt_det_wrt_whole_img


def detect_lmk_with_respect_to_crop(detector, all_img_paths, gt_detections, output_imgs, output_txt, num_total_gt_dets, path_file_no_face_detected):
    count_no_find_face = 0
    idx_bbox_global = 0
    for idx_img, img_path in enumerate(all_img_paths):
        print(f'Img {idx_img}/{len(all_img_paths)} - Reading {img_path} ...')
        img_bgr = cv2.imread(img_path)

        img_file_name = os.path.basename(img_path)
        gt_dets_img = gt_detections[img_file_name]

        for idx_gt_det, gt_det in enumerate(gt_dets_img):
            start_time = time.time()
            img_bgr_crop, new_gt_det, new_gt_det_wrt_whole_img = crop_gt_bbox(img_bgr, gt_det, percent_expand=0.4)
            
            det_bboxes, det_points = detector.detect(img_bgr_crop, args.thresh, args.scales, do_flip=False)

            subj_id = str(new_gt_det['SUBJECT_ID'])
            path_subj_img = os.path.join(output_imgs, subj_id)
            os.makedirs(path_subj_img, exist_ok=True)
            path_subj_txt = os.path.join(output_txt, subj_id)
            os.makedirs(path_subj_txt, exist_ok=True)

            crop_file_name, crop_ext = img_file_name.split('.')
            crop_file_name += '_crop' + str(idx_gt_det).zfill(3) + '.png'
            path_img_bgr_crop = os.path.join(path_subj_img, crop_file_name)

            lmk_file_format_uccsV1_name = crop_file_name.split('.')[0] + '_format_uccsV1.txt'
            path_img_lmk_format_uccsV1 = os.path.join(path_subj_txt, lmk_file_format_uccsV1_name)

            lmk_file_format_uccsV2_name = crop_file_name.split('.')[0] + '_format_uccsV2.txt'
            path_img_lmk_format_uccsV2 = os.path.join(path_subj_txt, lmk_file_format_uccsV2_name)

            print(f'    idx_bbox_global {idx_bbox_global}/{num_total_gt_dets} - gt_det {idx_gt_det}/{len(gt_dets_img)}')

            if det_bboxes.shape[0] == 0:
                print(f'NO FACE DETECTED IN CROP \'{path_img_bgr_crop}\'')
                print(f'Adding path to file \'{path_file_no_face_detected}\' ...')
                add_string_end_file(path_file_no_face_detected, path_img_bgr_crop)
                count_no_find_face += 1

                new_bbox = (new_gt_det['FACE_X'], new_gt_det['FACE_Y'], new_gt_det['FACE_WIDTH'], new_gt_det['FACE_HEIGHT'])
                synth_points = get_generic_lmk(new_bbox)
                det_points = synth_points

            if args.draw_bbox_lmk:
                new_bbox = (new_gt_det['FACE_X'], new_gt_det['FACE_Y'], new_gt_det['FACE_WIDTH'], new_gt_det['FACE_HEIGHT'])
                img_bgr_crop = draw_bbox(img_bgr_crop, new_bbox, color=(0,255,0), thickness=6)
                img_bgr_crop = draw_lmks(img_bgr_crop, det_points.reshape((5, 2)))

            # SAVE CROPPED FACE
            print(f'    Saving crop \'{path_img_bgr_crop}\'')
            cv2.imwrite(path_img_bgr_crop, img_bgr_crop)

            # SAVE DETECTIONS AS TXT FILE IN FORMAT UCCSv1
            new_gt_det['IMAGE'] = crop_file_name
            detections_uccsV1 = {crop_file_name: [([new_gt_det], det_points)]}
            print(f'    Saving {path_img_lmk_format_uccsV1} ...')
            save_detections_txt_format_uccsV1(detections_uccsV1, path_img_lmk_format_uccsV1)

            # SAVE DETECTIONS AS TXT FILE IN FORMAT UCCSv2
            confidences = [1.0]
            new_bbox = [(new_gt_det['FACE_X'], new_gt_det['FACE_Y'], new_gt_det['FACE_WIDTH'], new_gt_det['FACE_HEIGHT'])]
            detections_uccsV2 = {crop_file_name: [(confidences, new_bbox, det_points)]}
            print(f'    Saving {path_img_lmk_format_uccsV2} ...')
            save_detections_txt_format_uccsV2(detections_uccsV2, path_img_lmk_format_uccsV2)

            idx_bbox_global += 1

            elapsed_time = time.time() - start_time
            print(f'    Elapsed time: {elapsed_time} seconds')
            print(f'    {count_no_find_face} crops without faces (paths saved in \'{path_file_no_face_detected}\')')
            print('    ------')
            
            # if idx_bbox_global == 10: sys.exit(0)

        print('---------------------')
    return count_no_find_face


def detect_lmk_keep_whole_image(detector, all_img_paths, gt_detections, output_imgs, output_txt, num_total_gt_dets, path_file_no_face_detected):
    count_no_find_face = 0
    idx_bbox_global = 0
    for idx_img, img_path in enumerate(all_img_paths):
        print(f'Img {idx_img}/{len(all_img_paths)} - Reading {img_path} ...')
        img_bgr = cv2.imread(img_path)

        img_file_name = os.path.basename(img_path)
        gt_dets_img = gt_detections[img_file_name]
        detections_uccsV1 = {}
        detections_uccsV2 = {}

        det_lmk_one_face_crop_uccsV1 = []
        det_lmk_one_face_crop_uccsV2 = []

        img_bgr_with_bbox_lmks = None
        for idx_gt_det, gt_det in enumerate(gt_dets_img):
            start_time = time.time()
            print(f'    Cropping ground truth face bbox...')
            img_bgr_crop, new_gt_det, new_gt_det_wrt_whole_img = crop_gt_bbox(img_bgr, gt_det, percent_expand=0.4)
            
            print(f'    idx_bbox_global {idx_bbox_global}/{num_total_gt_dets} - gt_det {idx_gt_det}/{len(gt_dets_img)}')
            print(f'    Detecting face...')
            det_bboxes, det_points = detector.detect(img_bgr_crop, args.thresh, args.scales, do_flip=False)

            if det_bboxes.shape[0] > 0:
                # Keep only 1 face per gt bbox and adjust landmarks coords with respect to whole image
                det_points = np.expand_dims(det_points[0], axis=0)
                for idx_det_point, det_point in enumerate(det_points[0]):
                    det_point[0] += new_gt_det_wrt_whole_img['FACE_Y']
                    det_point[1] += new_gt_det_wrt_whole_img['FACE_X']
            else: # if no face was detected
                crop_file_name = img_file_name + ', FACE_ID=' + str(gt_det['FACE_ID']) + ', SUBJECT_ID=' + str(gt_det['SUBJECT_ID'])
                print(f'    NO FACE DETECTED IN CROP \'{crop_file_name}\'')
                print(f'    Adding path to file \'{path_file_no_face_detected}\' ...')
                add_string_end_file(path_file_no_face_detected, crop_file_name)
                count_no_find_face += 1

                new_bbox = (gt_det['FACE_X'], gt_det['FACE_Y'], gt_det['FACE_WIDTH'], gt_det['FACE_HEIGHT'])
                synth_points = get_generic_lmk(new_bbox)
                det_points = synth_points

            if args.draw_bbox_lmk:
                if img_bgr_with_bbox_lmks is None: img_bgr_with_bbox_lmks = img_bgr.copy()
                new_bbox = (gt_det['FACE_X'], gt_det['FACE_Y'], gt_det['FACE_WIDTH'], gt_det['FACE_HEIGHT'])
                img_bgr_with_bbox_lmks = draw_bbox(img_bgr_with_bbox_lmks, new_bbox, color=(0,255,0), thickness=6)
                img_bgr_with_bbox_lmks = draw_lmks(img_bgr_with_bbox_lmks, det_points.reshape((5, 2)))

            gt_det['IMAGE'] = img_file_name
            det_lmk_one_face_crop_uccsV1.append(([gt_det], det_points))
            
            confidences = [1.0]
            new_bbox = [(gt_det['FACE_X'], gt_det['FACE_Y'], gt_det['FACE_WIDTH'], gt_det['FACE_HEIGHT'])]
            det_lmk_one_face_crop_uccsV2.append((confidences, new_bbox, det_points))

            idx_bbox_global += 1

            elapsed_time = time.time() - start_time
            print(f'    Elapsed time: {elapsed_time} seconds')
            print(f'    {count_no_find_face} crops without faces (paths saved in \'{path_file_no_face_detected}\')')
            print('    ------')
        
        lmk_file_format_uccsV1_name = img_file_name.split('.')[0] + '_format_uccsV1.txt'
        path_img_lmk_format_uccsV1 = os.path.join(output_txt, lmk_file_format_uccsV1_name)

        lmk_file_format_uccsV2_name = img_file_name.split('.')[0] + '_format_uccsV2.txt'
        path_img_lmk_format_uccsV2 = os.path.join(output_txt, lmk_file_format_uccsV2_name)

        # SAVE DETECTIONS AS TXT FILE IN FORMAT UCCSv1
        detections_uccsV1[img_file_name] = det_lmk_one_face_crop_uccsV1
        print(f'    Saving {path_img_lmk_format_uccsV1} ...')
        save_detections_txt_format_uccsV1(detections_uccsV1, path_img_lmk_format_uccsV1)

        # SAVE DETECTIONS AS TXT FILE IN FORMAT UCCSv2
        detections_uccsV2[img_file_name] = det_lmk_one_face_crop_uccsV2
        print(f'    Saving {path_img_lmk_format_uccsV2} ...')
        save_detections_txt_format_uccsV2(detections_uccsV2, path_img_lmk_format_uccsV2)

        if args.draw_bbox_lmk:
            # SAVE WHOLE IMAGE
            path_whole_img_with_bbox_lmks = os.path.join(output_imgs, img_file_name)
            print(f'    Saving whole image \'{path_whole_img_with_bbox_lmks}\'')
            cv2.imwrite(path_whole_img_with_bbox_lmks, img_bgr_with_bbox_lmks)

        print('---------------------')
    return count_no_find_face


def main_detect_lmk(args):
    print(f'Loading groundtruth detections \'{args.gt_path}\'')
    gt_detections, num_total_gt_dets = load_gt_detections(args.gt_path)

    input_dir = args.input_path.rstrip('/')
    if not os.path.exists(input_dir):
        print(f'The input path doesn\'t exists: {input_dir}')
        sys.exit(0)

    output_dir = args.output_path.rstrip('/')
    if output_dir == '':
        output_dir = input_dir + '_bbox_lmks_retinaface'
    os.makedirs(output_dir, exist_ok=True)

    output_imgs = os.path.join(output_dir.rstrip('/'), 'imgs')
    os.makedirs(output_imgs, exist_ok=True)

    output_txt = os.path.join(output_dir.rstrip('/'), 'txt')
    os.makedirs(output_txt, exist_ok=True)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file_no_face_detected = f'files_no_face_detected_thresh={args.thresh}_starttime={formatted_datetime}.txt'
    path_file_no_face_detected = os.path.join(output_dir, file_no_face_detected)

    det_path = './retinaface/model/retinaface-R50/R50'
    print(f'\nLoading face detector \'{det_path}\'...')
    detector = RetinaFace(det_path, 0, args.gpu, 'net3')

    ext = args.input_ext
    all_img_paths = []
    if os.path.isdir(input_dir):
        print(f'\nSearching \'{ext}\' files with pattern \'{args.str_pattern}\' in path \'{input_dir}\' ...')
        all_img_paths = get_all_files_in_path(input_dir, ext, args.str_pattern)    

    assert len(all_img_paths) > 0, f'No files found with extention \'{ext}\' and pattern \'{args.str_pattern}\' in input \'{input_dir}\''
    print(f'{len(all_img_paths)} files found\n')

    if not args.coords_with_respect_to_crop:
        count_no_find_face = detect_lmk_keep_whole_image(detector, all_img_paths, gt_detections, output_imgs, output_txt, num_total_gt_dets, path_file_no_face_detected)
    else:
        count_no_find_face = detect_lmk_with_respect_to_crop(detector, all_img_paths, gt_detections, output_imgs, output_txt, num_total_gt_dets, path_file_no_face_detected)

    print('-------------------------------')
    print('Finished')
    print(f'{count_no_find_face}/{len(all_img_paths)} images without faces.')
    if count_no_find_face > 0:
        print(f'   Check in \'{path_file_no_face_detected}\'')


if __name__ == '__main__':
    args = getArgs()
    main_detect_lmk(args)
