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
import rawpy


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/3rd_OpensetFDIC_IJCB2024/validation_images', help='the dir your dataset of face which need to crop')
    parser.add_argument('--input_ext', type=str, default='jpg,png,jpeg,nef', help='jpg or png or jpeg or nef or jpg,png or jpg,png,jpeg')
    parser.add_argument('--output_ext', type=str, default='png', help='jpg or png or jpeg or nef or jpg,png or jpg,png,jpeg')
    parser.add_argument('--output_path', type=str, default='', help='the dir the cropped faces of your dataset where to save')

    parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--div', default=1, type=int, help='Number of parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    args = parser.parse_args()
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


def get_all_files_in_path(folder_path, file_extension=['.jpg','.png'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
    file_list.sort()
    return file_list


def convert_imgs_files(args):
    input_dir = args.input_path.rstrip('/')
    if not os.path.exists(input_dir):
        print(f'The input path doesn\'t exists: {input_dir}')
        sys.exit(0)

    output_dir = args.output_path.rstrip('/')
    if output_dir == '':
        output_dir = input_dir + f'_format={args.output_ext.upper()}'
    os.makedirs(output_dir, exist_ok=True)

    ext = args.input_ext.split(',')
    all_img_paths = []
    if os.path.isdir(input_dir):
        print(f'\nSearching files {ext} with pattern \'{args.str_pattern}\' in path \'{input_dir}\' ...')
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

    img_paths_part = img_paths_part[begin_index_str:end_index_str]
    for i, input_img_path in enumerate(img_paths_part):
        start_time = time.time()
        print(f'divs: {args.div}    part: {args.part}    files: {len(img_paths_part)}')
        print(f'begin_parts: {begin_parts}')
        print(f'  end_parts: {end_parts}')

        print(f'Img {i+1}/{len(img_paths_part)} - Reading \'{input_img_path}\'')
        # face_img = cv2.imread(input_img_path)
        if input_img_path.endswith('.nef'):
            raw_img = rawpy.imread(input_img_path)
            face_img = raw_img.postprocess()
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            face_img = cv2.imread(input_img_path)
        # sys.exit(0)


        output_img_path = input_img_path.replace(args.input_path, output_dir)
        output_img_path_split = output_img_path.split('/')
        input_img_name, input_img_ext = os.path.splitext(output_img_path_split[-1])
        output_img_path = os.path.join('/'.join(output_img_path_split[:-1]), f"{input_img_name}.{args.output_ext.strip('.')}")
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        print('output_img_path:', output_img_path)
        cv2.imwrite(output_img_path, face_img)
        # sys.exit(0)


        elapsed_time_sample = time.time() - start_time
        estimated_time_to_finish = elapsed_time_sample * (len(img_paths_part)-i+1)
        print(f'Elapsed time sample: {elapsed_time_sample} seconds')
        print('estimated_time_to_finish: %.2fs, %.2fm, %.2fh' % (estimated_time_to_finish, estimated_time_to_finish/60, estimated_time_to_finish/3600))
        print('-------------')


    print('-------------------------------')
    print('Finished')



if __name__ == '__main__':
    args = getArgs()
    convert_imgs_files(args)
