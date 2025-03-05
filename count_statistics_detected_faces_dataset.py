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
import matplotlib.pyplot as plt


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/txt')
    parser.add_argument('--input_ext', type=str, default='.txt', help='')
    parser.add_argument('--output_path', type=str, default='', help='the dir the cropped faces of your dataset where to save')
    # parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    # parser.add_argument('--face_size', type=int, default=112, help='the size of the face to save, the size x%2==0, and width equal height')
    # parser.add_argument('--thresh', type=float, default=0.01, help='threshold for face detection')
    # parser.add_argument('--nms', type=float, default=0.4, help='Non-Maximum Suppression')
    # parser.add_argument('--scales', type=str, default='[1.0]', help='the scale to resize image before detecting face')
    # parser.add_argument('--draw_bbox_lmk_save_whole_img', action='store_true', help='')
    # parser.add_argument('--save_crops', action='store_true', help='')
    # parser.add_argument('--align_face', action='store_true', help='')
    # parser.add_argument('--force_lmk', action='store_true', help='')

    parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--div', default=1, type=int, help='Number of parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    args = parser.parse_args()

    # args.scales = ast.literal_eval(args.scales)
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


def draw_bbox(img, bbox):
    result_img = img.copy()
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(result_img, (int(round(x1)), int(round(y1))),
                              (int(round(x2)), int(round(y2))), color, thickness)
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
def save_detections_txt(detections, saving_path):
    with open(saving_path, "w") as f:
        f.write("FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y\n")
        for image in sorted(detections.keys()):
            # for (score,bbox,lmark) in detections[image]:
            for (scores,bboxes,lmarks) in detections[image]:
                for bbox_idx in range(len(bboxes)):
                    score = scores[bbox_idx]
                    bbox = bboxes[bbox_idx]
                    lmark = lmarks[bbox_idx]
                    # f.write("%s,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n" % (image, score,
                    f.write("%s,%3.20f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n" % (image, score,
                                                                                        bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1],
                                                                                        lmark[0][0],lmark[0][1],
                                                                                        lmark[1][0],lmark[1][1],
                                                                                        lmark[2][0],lmark[2][1],
                                                                                        lmark[3][0],lmark[3][1],
                                                                                        lmark[4][0],lmark[4][1]))


def load_face_data_from_csv(file_path):
    data_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # Initialize the dictionary with empty lists for each header
        for header in csv_reader.fieldnames:
            data_dict[header] = []

        # Populate the dictionary with data from each row
        for row in csv_reader:
            for header in csv_reader.fieldnames:
                # Convert values to float where applicable
                try:
                    data_dict[header].append(float(row[header]))
                except ValueError:
                    data_dict[header].append(row[header])

    return data_dict


# def plot_face_dimensions_distribution(widths, heights, output_file):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(widths, heights, alpha=0.6, edgecolor='k', label='Face Dimensions')
#     plt.title('Distribution of Face Widths and Heights')
#     plt.xlabel('Width')
#     plt.ylabel('Height')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.savefig(output_file, format='png')
#     plt.close()

def plot_face_dimensions_distribution(widths, heights, output_file):
    plt.figure(figsize=(10, 6))

    # Plot histograms with density curves
    bins = 30
    plt.hist(widths, bins=bins, alpha=0.5, color='red', label='Widths', density=True, edgecolor='black')
    plt.hist(heights, bins=bins, alpha=0.5, color='blue', label='Heights', density=True, edgecolor='black')

    plt.title('Distribution of Face Widths and Heights')
    plt.xlabel('Measure Value')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(output_file, format='png')
    plt.close()


def count_faces(args):
    input_dir = args.input_path.rstrip('/')
    if not os.path.exists(input_dir):
        print(f'The input path doesn\'t exists: {input_dir}')
        sys.exit(0)

    output_dir = args.output_path.rstrip('/')
    if output_dir == '':
        output_dir = input_dir + '/../statistics'
    # output_dir += f"_scales={str(args.scales).replace(' ','')}"
    # output_dir += f"_nms={args.nms}"
    os.makedirs(output_dir, exist_ok=True)
    # sys.exit(0)

    # output_imgs = os.path.join(output_dir.rstrip('/'), 'imgs')
    # os.makedirs(output_imgs, exist_ok=True)

    # output_txt = os.path.join(output_dir.rstrip('/'), 'txt')
    # os.makedirs(output_txt, exist_ok=True)

    # if args.gpu == -1:
    #     ctx = mx.cpu()
    # else:
    #     ctx = mx.gpu(args.gpu)
    # sys.exit(0)

    # current_datetime = datetime.datetime.now()
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # file_no_face_detected = f'files_no_face_detected_thresh={args.thresh}_starttime={formatted_datetime}.txt'
    # path_file_no_face_detected = os.path.join(output_dir, file_no_face_detected)

    # det_path = './retinaface/model/retinaface-R50/R50'
    # print(f'\nLoading face detector \'{det_path}\'...')
    # detector = RetinaFace(det_path, 0, args.gpu, 'net3', nms=args.nms)

    # count_no_find_face = 0
    # count_crop_images = 0

    ext = args.input_ext
    all_img_paths = []
    if os.path.isdir(input_dir):
        print(f'\nSearching \'{ext}\' files with pattern \'{args.str_pattern}\' in path \'{input_dir}\' ...')
        all_img_paths = get_all_files_in_path(input_dir, ext, args.str_pattern)    

    assert len(all_img_paths) > 0, f'No files found with extention \'{ext}\' and pattern \'{args.str_pattern}\' in input \'{input_dir}\''
    print(f'{len(all_img_paths)} files found')
    begin_parts, end_parts = get_parts_indices(all_img_paths, args.div)
    img_paths_part = all_img_paths[begin_parts[args.part]:end_parts[args.part]]

    begin_index_str = 0
    end_index_str = len(img_paths_part)

    if args.str_begin != '':
        print('\nSearching str_begin \'' + args.str_begin + '\' ...  ')
        for x, img_path in enumerate(img_paths_part):
            if args.str_begin in img_path:
                begin_index_str = x
                print('found at', begin_index_str)
                break

    if args.str_end != '':
        print('\nSearching str_end \'' + args.str_end + '\' ...  ')
        for x, img_path in enumerate(img_paths_part):
            if args.str_end in img_path:
                end_index_str = x+1
                print('found at', begin_index_str)
                break
    
    print('\n------------------------')
    print('begin_index_str:', begin_index_str)
    print('end_index_str:', end_index_str)
    print('------------------------\n')

    all_faces_data = [None] * len(img_paths_part)

    img_paths_part = img_paths_part[begin_index_str:end_index_str]
    for i, input_path_path in enumerate(img_paths_part):
        # start_time = time.time()
        # print(f'divs: {args.div}    part: {args.part}    files: {len(img_paths_part)}')
        # print(f'begin_parts: {begin_parts}')
        # print(f'  end_parts: {end_parts}')

        print(f'File {i+1}/{len(img_paths_part)} - Reading {input_path_path} ...')
        # face_img = cv2.imread(input_path_path)

        faces_data_one_file = load_face_data_from_csv(input_path_path)
        # print('faces_data_one_file:', faces_data_one_file)
        all_faces_data[i] = faces_data_one_file
        # sys.exit(0)

    # print('all_faces_data:', all_faces_data)
    # sys.exit(0)
    
    ALL_BB_WIDTH, ALL_BB_HEIGHT = [], []
    for idx_file in range(len(all_faces_data)):
        # print(f'{idx_file} - FILE:', all_faces_data[idx_file]['FILE'])
        one_file_data = all_faces_data[idx_file]
        for idx_face in range(len(one_file_data['FILE'])):
            # print(f"{idx_face} - FILE: {one_file_data['FILE']}")
            ALL_BB_WIDTH.append(one_file_data['BB_WIDTH'][idx_face])
            ALL_BB_HEIGHT.append(one_file_data['BB_HEIGHT'][idx_face])
    
    # print('len(ALL_BB_WIDTH):', len(ALL_BB_WIDTH))
    # print('len(ALL_BB_HEIGHT):', len(ALL_BB_HEIGHT))
    # print('ALL_BB_WIDTH:', ALL_BB_WIDTH)

    chart_file_path = output_dir + '/widths_and_heights_dist.png'
    plot_face_dimensions_distribution(ALL_BB_WIDTH, ALL_BB_HEIGHT, chart_file_path)
    print('Finished!')


if __name__ == '__main__':
    args = getArgs()
    count_faces(args)
