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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery_detect_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/gallery.csv', help='')
    parser.add_argument('--validation_detect_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/validation.csv', help='')
    parser.add_argument('--validation_detect_pred', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]/all_detections_thresh=0.01.txt', help='')

    parser.add_argument('--gallery_embedd_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/2D_embeddings/gallery_ground_truth', help='')
    parser.add_argument('--validation_embedd_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/2D_embeddings/validation_ground_truth', help='')
    parser.add_argument('--validation_embedd_detect_pred', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/2D_embeddings/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]_all_detections_thresh=0.01', help='')
    parser.add_argument('--embedd_ext', type=str, default='.npy', help='')

    parser.add_argument('--output_dir', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/analysis_dataset', help='the dir where to save results')
    # parser.add_argument('--scale-to-save', type=float, default=1.0, help='')
    # parser.add_argument('--start-string', type=str, default='', help='')
    # parser.add_argument('--save-images', action='store_true')

    args = parser.parse_args()
    return args


def get_all_files_in_path(folder_path, file_extension='.jpg', pattern=''):
    file_list = []
    num_found_files = 0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            if pattern in path_file and path_file.endswith(file_extension):
                file_list.append(path_file)
                num_found_files += 1
                print(f'    num_found_files {num_found_files}', end='\r')
    print('')
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
                    if headers[i] == 'FILE' or headers[i] == 'SUBJECT_ID' or headers[i] == 'TYPE':
                        row_dict[headers[i]] = value
                    else:
                        row_dict[headers[i]] = float(value)
                data.append(row_dict)
    except Exception as e:
        print("Error loading CSV file:", e)
    return detect_set, data


def load_2D_embeddings(path, file_ext, normalize=True):
    print(f'    Searching files paths in \'{path}\'')
    embedds_paths = get_all_files_in_path(path, file_extension=file_ext)
    print(f'    Found {len(embedds_paths)} files')
    # print('embedds_paths:', embedds_paths)

    embedds_data = np.zeros((len(embedds_paths),512), dtype=np.float32)
    for idx_embedd_path, embedd_path in enumerate(embedds_paths):
        print(f'    Loading embeddings {idx_embedd_path}/{len(embedds_paths)}', end='\r')
        one_embedd = np.load(embedd_path)
        if normalize:
            one_embedd = one_embedd / np.linalg.norm(one_embedd)
        embedds_data[idx_embedd_path] = one_embedd
    print('')
    return embedds_data


def load_2D_embeddings_with_labels(path, file_ext, detections=None, normalize=True):
    print(f'    Searching files paths in \'{path}\'')
    embedds_paths = [None] * len(detections)
    # embedds_labels = [None] * len(detections)
    embedds_labels = -np.ones((len(detections),), dtype=int)
    for idx_det, det in enumerate(detections):
        print(f'    {idx_det}/{len(detections)}', end='\r')
        det_file = det['FILE']
        det_file_name, det_file_ext = os.path.splitext(det_file)
        detection_score = det['DETECTION_SCORE']
        label = det['TYPE']

        img_embedd_pattern = os.path.join(path, det_file_name, det_file_name+'_*_conf%.2f'%(detection_score)+'*'+file_ext)
        img_embedd_pattern = img_embedd_pattern.replace(']','*')
        # img_embedd_pattern = img_embedd_pattern.replace('.00','.0')
        img_embedd_pattern = img_embedd_pattern.replace('0*','*')
        # embedds_paths = get_all_files_in_path(path, file_extension=file_ext)
        # print('img_embedd_pattern:', img_embedd_pattern)
        embedd_path = glob.glob(img_embedd_pattern)
        assert len(embedd_path) > 0, f'Error, no such file with pattern=\'{img_embedd_pattern}\''
        embedd_path = embedd_path[0]
        # print('embedd_path:', embedd_path)
        embedds_paths[idx_det] = embedd_path
        embedds_labels[idx_det] = 1 if label == 'tp' else 0
        # sys.exit(0)
    print('')
    print(f'    Found {len(embedds_paths)} files')
    # sys.exit(0)

    # print(f'    Searching files paths in \'{path}\'')
    # embedds_paths = get_all_files_in_path(path, file_extension=file_ext)
    # print(f'    Found {len(embedds_paths)} files')
    
    embedds_data = np.zeros((len(embedds_paths),512), dtype=np.float32)
    for idx_embedd_path, embedd_path in enumerate(embedds_paths):
        print(f'    Loading embeddings {idx_embedd_path}/{len(embedds_paths)}', end='\r')
        one_embedd = np.load(embedd_path)
        if normalize:
            one_embedd = one_embedd / np.linalg.norm(one_embedd)
        embedds_data[idx_embedd_path] = one_embedd
    print('')
    return embedds_data, embedds_labels


def compute_detections_statistics(detections):
    detect_stats = {}
    face_widths = np.zeros((len(detections),), dtype=np.float32)
    face_heights = np.zeros((len(detections),), dtype=np.float32)
    face_areas = np.zeros((len(detections),), dtype=np.float32)
    for idx_detect, detect in enumerate(detections):
        print(f'    Computing detection statistics {idx_detect}/{len(detections)}', end='\r')
        try:
            face_widths[idx_detect] = detect['FACE_WIDTH']
            face_heights[idx_detect] = detect['FACE_HEIGHT']
            face_areas[idx_detect] = detect['FACE_WIDTH'] * detect['FACE_HEIGHT']
        except KeyError:
            face_widths[idx_detect] = detect['BB_WIDTH']
            face_heights[idx_detect] = detect['BB_HEIGHT']
            face_areas[idx_detect] = detect['BB_WIDTH'] * detect['BB_HEIGHT']

    print('')
    detect_stats['face_widths'] = face_widths
    detect_stats['face_heights'] = face_heights
    detect_stats['face_areas'] = face_areas
    return detect_stats


def compute_cosine_similarity(array1, array2, normalize=True):
    if array1.shape[0] == 1:
        array1 = array1[0]
    if array2.shape[0] == 1:
        array2 = array2[0]

    if isinstance(array1, np.ndarray):
         array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
         array2 = torch.from_numpy(array2)
    
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(array1, array2)
    return cos_sim


def compute_cosine_distance(array1, array2, normalize=True):
    cos_sim = compute_cosine_similarity(array1, array2, normalize)
    cos_dist = 1.0 - cos_sim
    return cos_dist


def compute_similarity_between_embeddings(embedds, mean_embedd):
    embedds_sims = np.zeros((len(embedds),), dtype=np.float32)
    for idx_embedd in range(len(embedds)):
        print(f'    Computing cosine distances {idx_embedd}/{len(embedds)}', end='\r')
        embedd = embedds[idx_embedd]
        embedds_sims[idx_embedd] = compute_cosine_similarity(embedd, mean_embedd, normalize=True)
    print('')
    return embedds_sims


def compute_all_similarities_between_embeddings(embeddings1, embeddings2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(embeddings1, np.ndarray):
         embeddings1 = torch.from_numpy(embeddings1)
    if isinstance(embeddings2, np.ndarray):
         embeddings2 = torch.from_numpy(embeddings2)

    embeddings1 = embeddings1.to(device)
    embeddings2 = embeddings2.to(device)
    
    embeddings1_normalized = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_normalized = F.normalize(embeddings2, p=2, dim=1)

    try:
        dot_product = torch.mm(embeddings1_normalized, embeddings2_normalized.t())
    except torch.cuda.OutOfMemoryError:
        print(f'    Allocating matrix {len(embeddings1)}x{len(embeddings2)}...')
        dot_product = torch.zeros(len(embeddings1), len(embeddings2))
        for idx1 in range(len(embeddings1)):
            print(f'    Computing cosine similarities {idx1}/{len(embeddings1)}', end='\r')
            # print(f'    \nembeddings1_normalized[idx1].size():', embeddings1_normalized[idx1].size())
            dot_product[idx1] = torch.mm(torch.unsqueeze(embeddings1_normalized[idx1],0), embeddings2_normalized.t())
        print('')

    cosine_similarities = dot_product.cpu().detach().numpy()
    return cosine_similarities


def get_mins_maxs_array(array, axis=1):
    mins = array.min(axis=axis)
    maxs = array.max(axis=axis)
    return mins, maxs


def save_histograms(data=[], legend=[''], title='', save_path='', xlim=None, ylim=None, bar_colors=None, bar_transparencies=None, bins=(50), figsize=(10, 6)):
    num_histograms = len(data)

    if not bar_colors:
        # bar_colors = ['b'] * num_histograms
        bar_colors = ['g', 'b']

    if not bar_transparencies:
        # bar_transparencies = [0.5] * num_histograms
        bar_transparencies = [0.5, 0.5]

    if len(data) != len(legend) or len(data) != len(bar_colors) or len(data) != len(bar_transparencies):
        print("Mismatch in lengths of data, legend, bar_colors, or bar_transparencies lists.")
        return

    plt.figure(figsize=figsize)

    for i in range(num_histograms):
        min_data = data[i].min()
        max_data = data[i].max()
        leg = legend[i] + ' (min=%.2f, max=%.2f)' % (min_data, max_data)
        plt.hist(data[i], alpha=bar_transparencies[i], color=bar_colors[i], label=leg, bins=bins[i])

    plt.legend()
    plt.title(title)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.savefig(save_path)
    plt.close()



def main(args):
    output_dir = args.output_dir.rstrip('/')
    print(f'Making output dir \'{output_dir}\'')
    os.makedirs(output_dir, exist_ok=True)
    print('Done\n')

    ################################

    print(f'Loading gallery groundtruth \'{args.gallery_detect_gt}\'')
    _, gallery_gt_detections = load_detections(args.gallery_detect_gt)
    # print('gallery_gt_detections:', gallery_gt_detections[0])
    print(f'    Loaded {len(gallery_gt_detections)} detections\n')
    # sys.exit(0)

    print(f'Loading validation groundtruth \'{args.validation_detect_gt}\'')
    _, validation_gt_detections = load_detections(args.validation_detect_gt)
    # print('validation_gt_detections:', validation_gt_detections[0])
    print(f'    Loaded {len(validation_gt_detections)} detections\n')
    # sys.exit(0)

    print(f'Loading validation detections \'{args.validation_detect_pred}\'')
    _, validation_pred_detections = load_detections(args.validation_detect_pred)
    # print('validation_pred_detections:', validation_pred_detections[0])
    print(f'    Loaded {len(validation_pred_detections)} detections\n')
    # sys.exit(0)

    print(f'Computing validation groundtruth detections statistics...')
    validation_gt_detections_stats = compute_detections_statistics(validation_gt_detections)
    print(f'Computing validation predictions detections statistics...')
    validation_pred_detections_stats = compute_detections_statistics(validation_pred_detections)

    hist_file_name = 'histogram_validation_face_widths.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[validation_pred_detections_stats['face_widths'], validation_gt_detections_stats['face_widths']],
                    legend=['validation preds face_widths', 'validation gt face_widths'],
                    title='Histograms - validation images',
                    # xlim=(0, 700),
                    bar_colors = ['b', 'g'],
                    bins=(100, 20),
                    save_path=hist_file_path)
    
    hist_file_name = 'histogram_validation_face_heights.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[validation_pred_detections_stats['face_heights'], validation_gt_detections_stats['face_heights']],
                    legend=['validation preds face_heights', 'validation gt face_heights'],
                    title='Histograms - validation images',
                    # xlim=(0, 700),
                    bar_colors = ['b', 'g'],
                    bins=(100, 20),
                    save_path=hist_file_path)
    
    hist_file_name = 'histogram_validation_face_areas.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[validation_pred_detections_stats['face_areas'], validation_gt_detections_stats['face_areas']],
                    legend=['validation preds face_areas', 'validation gt face_areas'],
                    title='Histograms - validation images',
                    # xlim=(0, 700),
                    bar_colors = ['b', 'g'],
                    bins=(100, 20),
                    save_path=hist_file_path)

    # sys.exit(0)
    ################################

    print(f'\nLoading gallery embeddings groundtruth \'{args.gallery_embedd_gt}\'')
    gallery_gt_embedds = load_2D_embeddings(args.gallery_embedd_gt, args.embedd_ext)
    print('    gallery_gt_embedds.shape:', gallery_gt_embedds.shape)
    gallery_gt_mean_embedd = gallery_gt_embedds.mean(axis=0)
    print('    gallery_gt_mean_embedd.shape:', gallery_gt_mean_embedd.shape)
    cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd = compute_similarity_between_embeddings(gallery_gt_embedds, gallery_gt_mean_embedd)
    print('    cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd.shape:', cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd.shape)
    # sys.exit(0)
    
    print(f'\nLoading validation embeddings groundtruth \'{args.validation_embedd_gt}\'')
    validation_gt_embedds = load_2D_embeddings(args.validation_embedd_gt, args.embedd_ext)
    print('    validation_gt_embedds.shape:', validation_gt_embedds.shape)
    validation_gt_mean_embedd = validation_gt_embedds.mean(axis=0)
    print('    validation_gt_mean_embedd.shape:', validation_gt_mean_embedd.shape)
    cos_sims_validationGtEmbedds_to_galleryGtMeanEmbedd = compute_similarity_between_embeddings(validation_gt_embedds, gallery_gt_mean_embedd)
    print('    cos_sims_validationGtEmbedds_to_galleryGtMeanEmbedd.shape:', cos_sims_validationGtEmbedds_to_galleryGtMeanEmbedd.shape)
    cos_sims_validationGtEmbedds_to_validationGtMeanEmbedd = compute_similarity_between_embeddings(validation_gt_embedds, validation_gt_mean_embedd)
    print('    cos_sims_validationGtEmbedds_to_validationGtMeanEmbedd.shape:', cos_sims_validationGtEmbedds_to_validationGtMeanEmbedd.shape)
    # sys.exit(0)

    print(f'\nLoading validation embeddings detections with labels \'{args.validation_embedd_detect_pred}\'')
    validation_pred_embedds, validation_pred_embedds_labels = load_2D_embeddings_with_labels(args.validation_embedd_detect_pred, args.embedd_ext, detections=validation_pred_detections)
    print('    validation_pred_embedds.shape:', validation_pred_embedds.shape)
    validation_pred_mean_embedd = validation_pred_embedds.mean(axis=0)
    print('    validation_pred_mean_embedd.shape:', validation_pred_mean_embedd.shape)
    cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd = compute_similarity_between_embeddings(validation_pred_embedds, gallery_gt_mean_embedd)
    print('    cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd.shape:', cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd.shape)
    cos_sims_validationPredEmbedds_to_validationGtMeanEmbedd = compute_similarity_between_embeddings(validation_pred_embedds, validation_gt_mean_embedd)
    print('    cos_sims_validationPredEmbedds_to_validationGtMeanEmbedd.shape:', cos_sims_validationPredEmbedds_to_validationGtMeanEmbedd.shape)
    # sys.exit(0)


    hist_file_name = 'histogram_gallery_validation_cosine_similarities.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd, cos_sims_validationGtEmbedds_to_galleryGtMeanEmbedd, cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd],
                    legend=['validationPredEmbedds_to_galleryGtMeanEmbedd', 'validationGtEmbedds_to_galleryGtMeanEmbedd', 'galleryGtEmbedds_to_galleryGtMeanEmbedd'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    ylim=(0, 30000),
                    bar_colors = ['r', 'b', 'g'],
                    bar_transparencies = (0.5, 0.5, 0.5),
                    bins=(50, 50, 50),
                    save_path=hist_file_path)


    hist_file_name = 'histogram_validation_validationGT_cosine_similarities.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[cos_sims_validationPredEmbedds_to_validationGtMeanEmbedd, cos_sims_validationGtEmbedds_to_validationGtMeanEmbedd, cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd],
                    legend=['validationPredEmbedds_to_validationGtMeanEmbedd', 'validationGtEmbedds_to_validationGtMeanEmbedd', 'galleryGtEmbedds_to_galleryGtMeanEmbedd'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    ylim=(0, 30000),
                    bar_colors = ['r', 'b', 'g'],
                    bar_transparencies = (0.5, 0.5, 0.5),
                    bins=(50, 50, 50),
                    save_path=hist_file_path)


    ##############################

    print(f'\nComputing cosine similarities between \'gallery_gt_embedds\' and \'gallery_gt_embedds\'...')
    cos_sims_galleryGtEmbedds_to_galleryGtEmbedds = compute_all_similarities_between_embeddings(gallery_gt_embedds, gallery_gt_embedds)
    print('    cos_sims_galleryGtEmbedds_to_galleryGtEmbedds.shape:', cos_sims_galleryGtEmbedds_to_galleryGtEmbedds.shape)
    mins_cos_sims_galleryGtEmbedds_to_galleryGtEmbedds, maxs_cos_sims_galleryGtEmbedds_to_galleryGtEmbedds = get_mins_maxs_array(cos_sims_galleryGtEmbedds_to_galleryGtEmbedds, axis=1)

    print(f'\nComputing cosine similarities between \'validation_gt_embedds\' and \'validation_gt_embedds\'...')
    cos_sims_validationGtEmbedds_to_validationGtEmbedds = compute_all_similarities_between_embeddings(validation_gt_embedds, validation_gt_embedds)
    print('    cos_sims_validationGtEmbedds_to_validationGtEmbedds.shape:', cos_sims_validationGtEmbedds_to_validationGtEmbedds.shape)
    mins_cos_sims_validationGtEmbedds_to_validationGtEmbedds, maxs_cos_sims_validationGtEmbedds_to_validationGtEmbedds = get_mins_maxs_array(cos_sims_validationGtEmbedds_to_validationGtEmbedds, axis=1)

    print(f'\nComputing cosine similarities between \'validation_gt_embedds\' and \'gallery_gt_embedds\'...')
    cos_sims_validationGtEmbedds_to_galleryGtEmbedds = compute_all_similarities_between_embeddings(validation_gt_embedds, gallery_gt_embedds)
    print('    cos_sims_validationGtEmbedds_to_galleryGtEmbedds.shape:', cos_sims_validationGtEmbedds_to_galleryGtEmbedds.shape)
    mins_cos_sims_validationGtEmbedds_to_galleryGtEmbedds, maxs_cos_sims_validationGtEmbedds_to_galleryGtEmbedds = get_mins_maxs_array(cos_sims_validationGtEmbedds_to_galleryGtEmbedds, axis=1)

    print(f'\nComputing cosine similarities between \'validation_pred_embedds\' and \'gallery_gt_embedds\'...')
    cos_sims_validationPredEmbedd_to_galleryGtEmbedds = compute_all_similarities_between_embeddings(validation_pred_embedds, gallery_gt_embedds)
    print('    cos_sims_validationPredEmbedd_to_galleryGtEmbedds.shape:', cos_sims_validationPredEmbedd_to_galleryGtEmbedds.shape)
    TP_cos_sims_validationPredEmbedd_to_galleryGtEmbedds = cos_sims_validationPredEmbedd_to_galleryGtEmbedds[validation_pred_embedds_labels==1]
    FP_cos_sims_validationPredEmbedd_to_galleryGtEmbedds = cos_sims_validationPredEmbedd_to_galleryGtEmbedds[validation_pred_embedds_labels==0]
    mins_cos_sims_validationPredEmbedd_to_galleryGtEmbedds, maxs_cos_sims_validationPredEmbedd_to_galleryGtEmbedds = get_mins_maxs_array(cos_sims_validationPredEmbedd_to_galleryGtEmbedds, axis=1)

    print(f'\nComputing cosine similarities between \'validation_pred_embedds\' and \'validation_gt_embedds\'...')
    cos_sims_validationPredEmbedd_to_validationGtEmbedds = compute_all_similarities_between_embeddings(validation_pred_embedds, validation_gt_embedds)
    print('    cos_sims_validationPredEmbedd_to_validationGtEmbedds.shape:', cos_sims_validationPredEmbedd_to_validationGtEmbedds.shape)
    TP_cos_sims_validationPredEmbedd_to_validationGtEmbedds = cos_sims_validationPredEmbedd_to_validationGtEmbedds[validation_pred_embedds_labels==1]
    FP_cos_sims_validationPredEmbedd_to_validationGtEmbedds = cos_sims_validationPredEmbedd_to_validationGtEmbedds[validation_pred_embedds_labels==0]
    mins_cos_sims_validationPredEmbedd_to_validationGtEmbedds, maxs_cos_sims_validationPredEmbedd_to_validationGtEmbedds = get_mins_maxs_array(cos_sims_validationPredEmbedd_to_validationGtEmbedds, axis=1)

    print('----------------')    

    hist_file_name = 'histogram_TP_FP_cosine_similarities_validationPred_to_galleryGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[FP_cos_sims_validationPredEmbedd_to_galleryGtEmbedds.flatten(), TP_cos_sims_validationPredEmbedd_to_galleryGtEmbedds.flatten()],
                    legend=['FP_validationPredEmbedd_to_galleryGtEmbedds', 'TP_validationPredEmbedd_to_galleryGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['r', 'g'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)
    
    hist_file_name = 'histogram_TP_FP_cosine_similarities_validationPred_to_validationGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[FP_cos_sims_validationPredEmbedd_to_validationGtEmbedds.flatten(), TP_cos_sims_validationPredEmbedd_to_validationGtEmbedds.flatten()],
                    legend=['FP_validationPredEmbedd_to_validationGtEmbedds', 'TP_validationPredEmbedd_to_validationGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['r', 'g'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)

    print('----------------')
    
    hist_file_name = 'histogram_min_max_cosine_similarities_galleryGt_galleryGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[mins_cos_sims_galleryGtEmbedds_to_galleryGtEmbedds, maxs_cos_sims_galleryGtEmbedds_to_galleryGtEmbedds],
                    legend=['mins_galleryGtEmbedds_to_galleryGtEmbedds', 'maxs_galleryGtEmbedds_to_galleryGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['g', 'b'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)

    hist_file_name = 'histogram_min_max_cosine_similarities_validationGt_to_validationGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[mins_cos_sims_validationGtEmbedds_to_validationGtEmbedds, maxs_cos_sims_validationGtEmbedds_to_validationGtEmbedds],
                    legend=['mins_validationGtEmbedds_to_validationGtEmbedds', 'maxs_validationGtEmbedds_to_validationGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['g', 'b'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)
    
    hist_file_name = 'histogram_min_max_cosine_similarities_validationGt_to_galleryGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[mins_cos_sims_validationGtEmbedds_to_galleryGtEmbedds, maxs_cos_sims_validationGtEmbedds_to_galleryGtEmbedds],
                    legend=['mins_validationGtEmbedds_to_galleryGtEmbedds', 'maxs_validationGtEmbedds_to_galleryGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['g', 'b'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)

    hist_file_name = 'histogram_min_max_cosine_similarities_validationPred_to_galleryGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[mins_cos_sims_validationPredEmbedd_to_galleryGtEmbedds, maxs_cos_sims_validationPredEmbedd_to_galleryGtEmbedds],
                    legend=['mins_validationPredEmbedd_to_galleryGtEmbedds', 'maxs_validationPredEmbedd_to_galleryGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['g', 'b'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)

    hist_file_name = 'histogram_min_max_cosine_similarities_validationPred_to_validationGt.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[mins_cos_sims_validationPredEmbedd_to_validationGtEmbedds, maxs_cos_sims_validationPredEmbedd_to_validationGtEmbedds],
                    legend=['mins_validationPredEmbedd_to_validationGtEmbedds', 'maxs_validationPredEmbedd_to_validationGtEmbedds'],
                    title='Histograms - Cosine similarities',
                    xlim=(-1.1, 1.1),
                    # ylim=(0, 30000),
                    bar_colors = ['g', 'b'],
                    bar_transparencies = (0.5, 0.5),
                    bins=(25, 25),
                    save_path=hist_file_path)


    print('\nFinished!\n')


if __name__ == '__main__':
    args = getArgs()
    main(args)