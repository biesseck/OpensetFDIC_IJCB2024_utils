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


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images', help='')
    parser.add_argument('--img_ext', type=str, default='.jpg', help='')
    parser.add_argument('--gt_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/validation.csv', help='')
    parser.add_argument('--pred_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5]/all_detections_thresh=0.01.txt', help='')
    parser.add_argument('--output_dir', type=str, default='', help='the dir where to save results')
    parser.add_argument('--iou', type=float, default=0.5, help='')
    parser.add_argument('--scale-to-save', type=float, default=1.0, help='')
    parser.add_argument('--start-string', type=str, default='', help='')

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


def count_tp_tn_fp_fn_img(gt_detections, pred_detections, iou_threshold):
    gt_detections_to_compare = copy.deepcopy(gt_detections)
    tp, fp, fn = 0, 0, 0
    for idx_pred, pred in enumerate(pred_detections):
        pred_bbox = (pred['BB_X'], pred['BB_Y'], pred['BB_WIDTH'], pred['BB_HEIGHT'])
        found_gt = False
        for idx_gt, gt in enumerate(gt_detections_to_compare):
            gt_bbox = (gt['FACE_X'], gt['FACE_Y'], gt['FACE_WIDTH'], gt['FACE_HEIGHT'])
            iou = calculate_iou(gt_bbox, pred_bbox)
            if iou >= iou_threshold:
                # tp += 1
                found_gt = True
                gt_detections_to_compare.remove(gt)
                break

        if found_gt:
            tp += 1
            pred['TYPE'] = 'tp'
        else:
            fp += 1
            pred['TYPE'] = 'fp'
    fn = len(gt_detections_to_compare)
    assert (tp+fp) == len(pred_detections)
    return tp, fp, fn


def eval_detections(gt_detections, pred_detections, iou_threshold=0.5):
    gt_keys = list(gt_detections.keys())
    pred_keys = list(pred_detections.keys())
    
    img_counts = {}
    for idx_gt, gt_img_name in enumerate(gt_keys):
        img_gt_det = gt_detections[gt_img_name]
        # print('gt_img_name:', gt_img_name, '    img_gt_det:', img_gt_det)
        num_gt_det = len(img_gt_det)
        num_pred_det = 0
        if gt_img_name in pred_keys:
            img_pred_det = pred_detections[gt_img_name]
            # print('gt_img_name:', gt_img_name, '    img_pred_det:', img_pred_det)
            num_pred_det = len(img_pred_det)

            tp, fp, fn = count_tp_tn_fp_fn_img(img_gt_det, img_pred_det, iou_threshold)
            img_counts[gt_img_name] = {'num_gt_det':num_gt_det, 'num_pred_det':num_pred_det, 'tp':tp, 'fp':fp, 'fn':fn}
            # print(f'img_counts[\'{gt_img_name}\']', img_counts[gt_img_name])
    # sys.exit(0)
    return img_counts


def draw_bbox(img, bbox, color, thickness=6):
    result_img = img.copy()
    # x1, y1, x2, y2 = bbox
    x1, y1, w, h = bbox
    # color = (0, 255, 0)  # Green color
    # thickness = 6
    cv2.rectangle(result_img, (int(round(x1)), int(round(y1))),
                              (int(round(x1+w)), int(round(y1+h))), color, thickness)
    return result_img


def draw_lmks(img, lmks, thickness=6):
    if not (type(lmks[0]) is list or type(lmks[0]) is np.ndarray):
        lmks = [lmks]
    result_img = img.copy()
    for l in range(len(lmks)):
        color = (0,0,255)  # red
        if l == 0 or l == 3:
            color = (255,191,0)  # blue
        cv2.circle(result_img, (int(round(lmks[l][0])), int(round(lmks[l][1]))), 1, color, thickness)
    return result_img


def save_imgs_with_detections(all_img_paths, gt_dets, pred_dets, detect_counts_per_img, output_dir, scale=1.0, start_idx=0):
    dir_save_imgs = 'imgs_with_bbox_lmk'
    output_dir = os.path.join(output_dir, dir_save_imgs)
    os.makedirs(output_dir, exist_ok=True)

    keys_gt_dets = list(gt_dets.keys())
    keys_pred_dets = list(pred_dets.keys())
    
    for idx_img, img_path in enumerate(all_img_paths):
        if idx_img < start_idx: continue

        img_file_name = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)

        img_gt_det = gt_dets[img_file_name]
        color_gt = (0, 255, 0)  # green
        for idx_gt, gt_det in enumerate(img_gt_det):
            gt_bbox = (gt_det['FACE_X'], gt_det['FACE_Y'], gt_det['FACE_WIDTH'], gt_det['FACE_HEIGHT'])
            img_bgr = draw_bbox(img_bgr, gt_bbox, color_gt, thickness=9)
        
        if img_file_name in keys_pred_dets:
            img_pred_det = pred_dets[img_file_name]
            for idx_pred, pred_det in enumerate(img_pred_det):
                pred_bbox = (pred_det['BB_X'], pred_det['BB_Y'], pred_det['BB_WIDTH'], pred_det['BB_HEIGHT'])
                pred_lmk = [[pred_det['REYE_X'], pred_det['REYE_Y']],
                            [pred_det['LEYE_X'], pred_det['LEYE_Y']],
                            [pred_det['NOSE_X'], pred_det['NOSE_Y']],
                            [pred_det['RMOUTH_X'], pred_det['RMOUTH_Y']],
                            [pred_det['LMOUTH_X'], pred_det['LMOUTH_Y']]]


                if pred_det['TYPE'] == 'tp':
                    color_pred = (255,191,0)  # blue
                elif pred_det['TYPE'] == 'fp':
                    color_pred = (0, 0, 255)  # red

                img_bgr = draw_bbox(img_bgr, pred_bbox, color_pred, thickness=6)
                img_bgr = draw_lmks(img_bgr, pred_lmk, thickness=10)
        
        path_output_img = os.path.join(output_dir, img_file_name)
        print(f'{idx_img}/{len(all_img_paths)} - Saving image with detections: \'{path_output_img}\'')

        if scale != 1.0:
            img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)

        cv2.imwrite(path_output_img, img_bgr)
    
    print('\nFinished!')






def main(args):
    input_dir = args.input_dir.rstrip('/')
    if not os.path.exists(input_dir):
        print(f'The input path doesn\'t exists: {input_dir}')
        sys.exit(0)

    output_dir = args.output_dir.rstrip('/')
    if output_dir == '':
        dir_results = f'analysis_results_scale-to-save={args.scale_to_save}'
        output_dir = os.path.join(os.path.dirname(args.pred_path), dir_results)

    print(f'Searching image files \'{input_dir}\'')
    all_img_paths = get_all_files_in_path(input_dir, file_extension=args.img_ext)
    print('    Found:', len(all_img_paths), 'files')
    # sys.exit(0)

    print(f'Loading groundtruth detections \'{args.gt_path}\'')
    gt_detections = load_gt_detections(args.gt_path)
    # print('gt_detections:', gt_detections)
    # print('gt_detections[\'00085e794b56df6e46a8a8b51cf23d71.jpg\']:', gt_detections['00085e794b56df6e46a8a8b51cf23d71.jpg'])
    # sys.exit(0)

    print(f'Loading pred detections \'{args.pred_path}\'')
    pred_detections = load_pred_detections(args.pred_path)
    # print('pred_detections[\'0006c02f81b58512e7a28059650b99f2.jpg\']:', pred_detections['0006c02f81b58512e7a28059650b99f2.jpg'])
    # print('len(pred_detections[\'0006c02f81b58512e7a28059650b99f2.jpg\']):', len(pred_detections['0006c02f81b58512e7a28059650b99f2.jpg']))
    # sys.exit(0)

    detect_counts_per_img = eval_detections(gt_detections, pred_detections, args.iou)
    # for idx_det_count, det_count_key in enumerate(detect_counts_per_img):
    #     print(f'{det_count_key}:', detect_counts_per_img[det_count_key])
    # sys.exit(0)
    
    start_idx = 0
    if args.start_string != '':
        for idx_img_path, img_path in enumerate(all_img_paths):
            if args.start_string in img_path:
                start_idx = idx_img_path

    os.makedirs(output_dir, exist_ok=True)
    save_imgs_with_detections(all_img_paths, gt_detections, pred_detections,
                              detect_counts_per_img, output_dir, args.scale_to_save, start_idx)


    # draw_gt_pred_detections(all_img_paths, gt_detections, pred_detections)

    '''
    print(f'Loading individual detections from \'{input_dir}\'')
    all_detections, num_det_faces = load_all_detections(all_img_paths)
    assert len(all_img_paths) == len(all_detections)
    print('    Loaded:', len(all_detections), 'files')
    print('    num_det_faces:', num_det_faces)

    output_detections_path = os.path.join(output_dir, f'all_detections_thresh={args.thresh}.txt')
    print(f'Saving merged file \'{output_detections_path}\'')
    num_saved_faces = save_detections_txt(all_detections, output_detections_path, args.thresh)
    print('    num_saved_faces:', num_saved_faces)

    print('Finished!\n')
    '''




if __name__ == '__main__':
    args = getArgs()
    main(args)