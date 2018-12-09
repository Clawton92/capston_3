import pandas as pd
import numpy as np

'''Start by editing description files
    1) Change all Benign_without_callback to Benign
    2) Find only unique instances of image path, cropped path and remove duplicates
    3) Possibly split files into CC and MLO?'''

def benign_callback_to_benign(x):
    if x == 'BENIGN_WITHOUT_CALLBACK':
        return 'BENIGN'
    else:
        return x

def split_val_path(x):
    ls = x.split('/')
    return ls[1]

def get_unique_csv(file_path, target_path=None, save=False):
    file = pd.read_csv(file_path)
    new_frame = file.drop('pathology', axis=1)
    new_frame['pathology'] = file['pathology'].apply(benign_callback_to_benign)
    new_frame['cropped image file path'] = file['cropped image file path'].apply(split_val_path)
    new_frame['image file path'] = file['image file path'].apply(split_val_path)
    new_frame['ROI mask file path'] = file['ROI mask file path'].apply(split_val_path)
    if save==True:
        new_frame.to_csv(target_path)
    else:
        return new_frame


if __name__=='__main__':

    mass_train_source = '/Users/christopherlawton/galvanize/module_2/capstone_2/original_csvs/mass_case_description_train_set.csv'
    mass_train_target = '/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/mass_train_unique_paths.csv'

    mass_test_source = '/Users/christopherlawton/galvanize/module_2/capstone_2/original_csvs/mass_case_description_test_set.csv'
    mass_test_target = '/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/mass_test_unique_paths.csv'

    calc_train_source = '/Users/christopherlawton/galvanize/module_2/capstone_2/original_csvs/calc_case_description_train_set.csv'
    calc_train_target = '/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/calc_train_unique_paths.csv'

    calc_test_source = '/Users/christopherlawton/galvanize/module_2/capstone_2/original_csvs/calc_case_description_test_set.csv'
    calc_test_target = '/Users/christopherlawton/galvanize/module_2/capstone_2/unique_image_csvs/calc_test_unique_paths.csv'

    #save all csvs to get unique meta data identifier (image paths/UID) for structuring directories
    get_unique_csv(mass_train_source, mass_train_target, save=True)
    get_unique_csv(mass_test_source, mass_test_target, save=True)
    get_unique_csv(calc_train_source, calc_train_target, save=True)
    get_unique_csv(calc_test_source, calc_test_target, save=True)
