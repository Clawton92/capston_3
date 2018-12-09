import os,sys
import shutil
import pydicom
import pandas as pd
import numpy as np
# import cv2
import matplotlib.pyplot as plt

'''script for zero padding'''
import cv2
desired_size = 250
im_pth = path
im = cv2.imread(im_pth)
old_size = im.shape[:2] # old_size is in (height, width) format
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
# new_size should be in (width, height) format
im = cv2.resize(im, (new_size[1], new_size[0]))
delta_w = desired_size - new_size[1]
delta_h = desired_size - new_size[0]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)
color = [0]
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

def write_images(img, destination_path, scale=True):
    '''Scaling will convert pixel values to floats to avoid underflow
    or overflow losses. Rescale grayscale image bwteen 0-255 opposed to 0-65535.
    Lastly, convert the image to uint8 opposed to uint16'''

    if scale==True:
        shape = img.pixel_array.shape
        image_2d = img.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
        # cv2.imwrite('{}.png'.format(destination_path), image_2d_scaled)
    # else:
    #     cv2.imwrite('{}.png'.format(destionation_path), image_2d_scaled)




def structure_full_mam_dirs(root_directory, dataframe, target_path, scale=True):

    '''This function sturctures new dirs with original train test split
    only for full mammogram files using the image file path in the root csvs'''

    file_ls = []
    path_vals = []

    count = 0
    for dir,subdirs,files in sorted(os.walk(root_directory)):
        for file in files:
            fullpath = os.path.realpath( os.path.join(dir,file) )
            dirname = fullpath.split(os.path.sep)
            if dirname[-1][-3:] == 'dcm':
                ds = pydicom.read_file(fullpath)
                classification = dataframe[dataframe['image file path'] == ds.StudyInstanceUID]['pathology'].values[0]
                if dirname[4][-1] == 'C':
                    view = 'CC'
                else:
                    view = 'MLO'
                destination_path = '{}/{}/{}/{}'.format(target_path, view, classification, dirname[4])
                write_images(ds, destination_path, scale)
                count+=1



def structure_roi_cropped_dirs(root_directory, dataframe, roi_destination, cropped_destination, scale=True):

    count = 0
    for dir,subdirs,files in sorted(os.walk(root_directory)):
        for file in files:
            fullpath = os.path.realpath( os.path.join(dir,file) )
            dirname = fullpath.split(os.path.sep)
            if dirname[-1][-3:] == 'dcm':
                ds = pydicom.read_file(fullpath)
                if ds.SeriesDescription == 'cropped images':
                    classification = dataframe[dataframe['cropped image file path'] == ds.StudyInstanceUID]['pathology'].values[0]
                    if dirname[5][-3] == 'C':
                        view = 'CC'
                    else:
                        view = 'MLO'
                    destination_path = '{}/{}/{}/{}_crop'.format(cropped_destination, view, classification, dirname[5][:-2])
                elif ds.SeriesDescription == 'ROI mask images':
                    if dirname[5][-3] == 'C':
                        view = 'CC'
                    else:
                        view = 'MLO'
                    classification = dataframe[dataframe['ROI mask file path'] == ds.StudyInstanceUID]['pathology'].values[0]
                    destination_path = '{}/{}/{}/{}_roi'.format(roi_destination, view, classification, dirname[5][:-2])
                write_images(ds, destination_path, scale)




if __name__ == '__main__':

    #mass train
    mass_train = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/mass_train_unique_paths.csv')
    mass_train_unique = mass_train.drop_duplicates('image file path')

    mass_train_full_root = '/Volumes/chris_external_drive/Mass-train-CBIS-DDSM'
    mass_train_roi_crop_root = '/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM'

    mass_train_full_target = '/Volumes/chris_external_drive/all_structured_scaled_images/full_mammograms/mass_train'
    mass_train_crop_target = '/Volumes/chris_external_drive/all_structured_scaled_images/cropped_images/mass_train'
    mass_train_roi_target = '/Volumes/chris_external_drive/all_structured_scaled_images/roi_images/mass_train'


    #mass test
    mass_test = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/mass_test_unique_paths.csv')
    mass_test_unique = mass_test.drop_duplicates('image file path') #use only for full mammogram images
    #root dirs
    mass_test_full_root = '/Volumes/chris_external_drive/Mass-test-CBIS-DDSM'
    mass_test_roi_crop_root = '/Volumes/chris_external_drive/mass_test_ROI_crop/CBIS-DDSM'
    #target root dirs
    mass_test_full_target = '/Volumes/chris_external_drive/all_structured_scaled_images/full_mammograms/mass_test'
    mass_test_crop_target = '/Volumes/chris_external_drive/all_structured_scaled_images/cropped_images/mass_test'
    mass_test_roi_target = '/Volumes/chris_external_drive/all_structured_scaled_images/roi_images/mass_test'



    #calc train
    calc_train = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/calc_train_unique_paths.csv')
    calc_train_unique = calc_train.drop_duplicates('image file path') #use only for full mammogram images
    #root dirs
    calc_train_full_root = '/Volumes/chris_external_drive/Calc-train-CBIS-DDSM'
    calc_train_roi_crop_root = '/Volumes/chris_external_drive/calc_train_ROI_crop/CBIS-DDSM'
    #target root dirs
    calc_train_full_target = '/Volumes/chris_external_drive/all_structured_scaled_images/full_mammograms/calc_train'
    calc_train_crop_target = '/Volumes/chris_external_drive/all_structured_scaled_images/cropped_images/calc_train'
    calc_train_roi_target = '/Volumes/chris_external_drive/all_structured_scaled_images/roi_images/calc_train'


    #calc test
    calc_test = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/calc_test_unique_paths.csv')
    calc_test_unique = calc_test.drop_duplicates('image file path')#use only for full mammogram images
    #root dirs
    calc_test_full_root = '/Volumes/chris_external_drive/Calc-test-CBIS-DDSM'
    calc_test_roi_crop_root = '/Volumes/chris_external_drive/calc_test_ROI_crop/CBIS-DDSM'
    #target root dirs
    calc_test_full_target = '/Volumes/chris_external_drive/all_structured_scaled_images/full_mammograms/calc_test'
    calc_test_crop_target = '/Volumes/chris_external_drive/all_structured_scaled_images/cropped_images/calc_test'
    calc_test_roi_target = '/Volumes/chris_external_drive/all_structured_scaled_images/roi_images/calc_test'

    '''full'''
    # structure_full_mam_dirs(mass_train_full_root, mass_train_unique, mass_train_full_target, scale=True)
    # structure_full_mam_dirs(mass_test_full_root, mass_test_unique, mass_test_full_target, scale=True)
    # structure_full_mam_dirs(calc_train_full_root, calc_train_unique, calc_train_full_target, scale=True)
    # structure_full_mam_dirs(calc_test_full_root, calc_test_unique, calc_test_full_target, scale=True)

    '''roi/crop'''
    # structure_roi_cropped_dirs(mass_train_roi_crop_root, mass_train, mass_train_roi_target, mass_train_crop_target, scale=True)
    # structure_roi_cropped_dirs(mass_test_roi_crop_root, mass_test, mass_test_roi_target, mass_test_crop_target, scale=True)
    # structure_roi_cropped_dirs(calc_train_roi_crop_root, calc_train, calc_train_roi_target, calc_train_crop_target, scale=True)
    # structure_roi_cropped_dirs(calc_test_roi_crop_root, calc_test, calc_test_roi_target, calc_test_crop_target, scale=True)
