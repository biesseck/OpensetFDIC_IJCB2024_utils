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



def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery_detect_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/gallery.csv', help='')
    parser.add_argument('--validation_detect_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/protocols/validation.csv', help='')
    parser.add_argument('--validation_detect_pred', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]/all_detections_thresh=0.01.txt', help='')

    parser.add_argument('--gallery_embedd_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/2D_embeddings/gallery_ground_truth', help='')
    parser.add_argument('--validation_embedd_gt', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/2D_embeddings/validation_ground_truth', help='')
    parser.add_argument('--validation_embedd_detect_pred', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/2D_embeddings/validation_images_DETECTED_FACES_RETINAFACE_scales=[0.15,0.2,0.5,1.0,1.2,1.5]_all_detections_thresh=0.01', help='')
    parser.add_argument('--embedd_ext', type=str, default='.npy', help='')

    # parser.add_argument('--output_dir', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/analysis_dataset', help='the dir where to save results')
    # parser.add_argument('--scale-to-save', type=float, default=1.0, help='')
    # parser.add_argument('--start-string', type=str, default='', help='')
    # parser.add_argument('--save-images', action='store_true')

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


def save_histograms(data=[], legend=[''], title='', save_path='', xlim=None, ylim=None, bar_colors=None, bar_transparencies=None, bins=(50), figsize=(10, 6)):
    num_histograms = len(data)

    if not bar_colors:
        # bar_colors = ['b'] * num_histograms
        bar_colors = ['g', 'b']

    if not bar_transparencies:
        # bar_transparencies = [0.5] * num_histograms
        bar_transparencies = [0.9, 0.5]

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


def remove_items_by_indexes(lst, indexes):
    indexes = list(set(indexes))   # remove duplicated indexes
    indexes.sort(reverse=True)
    for index in indexes:
        if 0 <= index < len(lst):
            del lst[index]
    return lst


def filter_detections_by_size(gt_detections_stats, pred_detections):
    gt_detections_min_width, gt_detections_max_width =   gt_detections_stats['face_widths'].min(),  gt_detections_stats['face_widths'].max()
    gt_detections_min_height, gt_detections_max_height = gt_detections_stats['face_heights'].min(), gt_detections_stats['face_heights'].max()
    gt_detections_min_area, gt_detections_max_area =     gt_detections_stats['face_areas'].min(),   gt_detections_stats['face_areas'].max()
    print(f'    gt_detections_min_width: {gt_detections_min_width}, gt_detections_max_width: {gt_detections_max_width}')
    print(f'    gt_detections_min_height: {gt_detections_min_height}, gt_detections_max_height: {gt_detections_max_height}')
    print(f'    gt_detections_min_area: {gt_detections_min_area}, gt_detections_max_area: {gt_detections_max_area}')

    idxes_to_remove = []
    for idx_pred_detect, pred_detect in enumerate(pred_detections):
        print(f'    Checking {idx_pred_detect}/{len(pred_detections)} - Detections to remove: {len(idxes_to_remove)}', end='\r')
        try:
            pred_detect_width  = pred_detect['FACE_WIDTH']
            pred_detect_height = pred_detect['FACE_HEIGHT']
        except KeyError:
            pred_detect_width  = pred_detect['BB_WIDTH']
            pred_detect_height = pred_detect['BB_HEIGHT']
        finally:
            pred_detect_area = pred_detect_width * pred_detect_height
        
        if pred_detect_width < gt_detections_min_width or pred_detect_width > gt_detections_max_width or \
           pred_detect_height < gt_detections_min_height or pred_detect_height > gt_detections_max_height or \
           pred_detect_area < gt_detections_min_area or pred_detect_area > gt_detections_max_area:
            idxes_to_remove.append(idx_pred_detect)
    print('')
    return idxes_to_remove


def filter_detections_by_cosine_similarity(gt_detections_cos_sims, pred_detections_cos_sims):
    gt_detections_min_cos_sim, gt_detections_max_cos_sim =   gt_detections_cos_sims.min(),  gt_detections_cos_sims.max()
    print(f'    gt_detections_min_cos_sim: {gt_detections_min_cos_sim}, gt_detections_max_cos_sim: {gt_detections_max_cos_sim}')

    idxes_to_remove = []
    for idx_pred_detect, pred_detect_cos_sim in enumerate(pred_detections_cos_sims):
        print(f'    Checking {idx_pred_detect}/{len(pred_detections_cos_sims)} - Detections to remove: {len(idxes_to_remove)}', end='\r')
        if pred_detect_cos_sim < gt_detections_min_cos_sim:
            idxes_to_remove.append(idx_pred_detect)
    print('')
    return idxes_to_remove


# based on https://github.com/AIML-IfI/uccs-facerec-challenge/blob/main/facerec/face_detection.py
def save_detections_txt(detections, saving_path, thresh=0.0):
    num_saved_faces = 0
    with open(saving_path, "w") as f:
        header = "FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y"
        header_list = header.split(',')
        f.write(header + "\n")
        for d, detect in enumerate(detections):
            if detect['DETECTION_SCORE'] >= thresh:
                num_saved_faces += 1
                for k, key in enumerate(header_list):
                    if k == 0:
                        f.write("%s" % detect[key])
                    elif key == 'DETECTION_SCORE':
                        f.write(",%3.20f" % detect[key])
                    else:
                        f.write(",%3.2f" % detect[key])
                f.write("\n")
    return num_saved_faces



def main(args):
    output_dir = os.path.dirname(args.validation_detect_pred)
    
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

    '''
    hist_file_name = 'histogram_validation_face_widths.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[validation_gt_detections_stats['face_widths'], validation_pred_detections_stats['face_widths']],
                    legend=['validation gt face_widths', 'validation preds face_widths'],
                    title='Histograms - validation images',
                    # xlim=(0, 700),
                    bins=(20, 100),
                    save_path=hist_file_path)
    
    hist_file_name = 'histogram_validation_face_heights.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[validation_gt_detections_stats['face_heights'], validation_pred_detections_stats['face_heights']],
                    legend=['validation gt face_heights', 'validation preds face_heights'],
                    title='Histograms - validation images',
                    # xlim=(0, 700),
                    bins=(20, 100),
                    save_path=hist_file_path)
    
    hist_file_name = 'histogram_validation_face_areas.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[validation_gt_detections_stats['face_areas'], validation_pred_detections_stats['face_areas']],
                    legend=['validation gt face_areas', 'validation preds face_areas'],
                    title='Histograms - validation images',
                    # xlim=(0, 700),
                    bins=(20, 100),
                    save_path=hist_file_path)
    '''

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
    # sys.exit(0)

    print(f'\nLoading validation embeddings detections \'{args.validation_embedd_detect_pred}\'')
    validation_pred_embedds = load_2D_embeddings(args.validation_embedd_detect_pred, args.embedd_ext)
    print('    validation_pred_embedds.shape:', validation_pred_embedds.shape)
    validation_pred_mean_embedd = validation_pred_embedds.mean(axis=0)
    print('    validation_pred_mean_embedd.shape:', validation_pred_mean_embedd.shape)
    cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd = compute_similarity_between_embeddings(validation_pred_embedds, gallery_gt_mean_embedd)
    print('    cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd.shape:', cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd.shape)
    # sys.exit(0)


    '''
    hist_file_name = 'histogram_gallery_validation_cosine_similarities.png'
    hist_file_path = os.path.join(output_dir, hist_file_name)
    print(f'Saving chart \'{hist_file_path}\'')
    save_histograms(data=[cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd, cos_sims_validationGtEmbedds_to_galleryGtMeanEmbedd, cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd],
                    legend=['galleryGtEmbedds_to_galleryGtMeanEmbedd', 'validationGtEmbedds_to_galleryGtMeanEmbedd', 'validationPredEmbedds_to_galleryGtMeanEmbedd'],
                    title='Histograms - Cosine similarities',
                    # xlim=(0, 700),
                    bar_colors = ['g', 'b', 'r'],
                    bar_transparencies = (0.9, 0.5, 0.3),
                    bins=(50, 50, 50),
                    save_path=hist_file_path)
    '''

    print(f'\nFiltering detections by size...')
    idxes_to_remove_by_size = filter_detections_by_size(validation_gt_detections_stats, validation_pred_detections)

    print(f'\nFiltering detections cosine similarity...')
    idxes_to_remove_by_cosine_sim = filter_detections_by_cosine_similarity(cos_sims_galleryGtEmbedds_to_galleryGtMeanEmbedd, cos_sims_validationPredEmbedds_to_galleryGtMeanEmbedd)

    all_idxes_to_remove = idxes_to_remove_by_size + idxes_to_remove_by_cosine_sim
    print(f'\nRemoving {len(all_idxes_to_remove)} detections...')
    validation_pred_detections = remove_items_by_indexes(validation_pred_detections, all_idxes_to_remove)
    print(f'    Final size: {len(validation_pred_detections)} detections')

    filtered_detections_file_name = os.path.splitext(os.path.basename(args.validation_detect_pred))[0] + '_FILTERED_BY_SIZE_COSINE_SIMILARITY' + '.txt'
    filtered_detections_file_path = os.path.join(output_dir, filtered_detections_file_name)
    print(f'\nSaving filtered detections \'{filtered_detections_file_path}\'')
    save_detections_txt(validation_pred_detections, filtered_detections_file_path)
    print('Done')

    print('\nFinished!')


if __name__ == '__main__':
    args = getArgs()
    main(args)