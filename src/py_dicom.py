import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

full = pydicom.read_file('/Volumes/chris_external_drive/Mass-train-CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC/07-20-2016-DDSM-79148/1-full mammogram images-44331/000000.dcm')
roi_1 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC_1/07-21-2016-DDSM-75344/1-ROI mask images-25492/000000.dcm')
crop_1 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC_1/07-21-2016-DDSM-75344/1-ROI mask images-25492/000001.dcm')

roi_2 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC_2/07-21-2016-DDSM-33313/1-ROI mask images-60878/000000.dcm')
crop_2 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC_2/07-21-2016-DDSM-33313/1-ROI mask images-60878/000001.dcm')

roi_3 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC_3/07-21-2016-DDSM-14174/1-ROI mask images-14592/000000.dcm')
crop_3 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_CC_3/07-21-2016-DDSM-14174/1-ROI mask images-14592/000001.dcm')

mlo_roi_1 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_MLO_1/07-21-2016-DDSM-76443/1-ROI mask images-80862/000000.dcm')
mlo_crop_1 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_MLO_1/07-21-2016-DDSM-76443/1-ROI mask images-80862/000001.dcm')

mlo_roi_2 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_MLO_2/07-21-2016-DDSM-98278/1-ROI mask images-14207/000000.dcm')
mlo_crop_2 = pydicom.read_file('/Volumes/chris_external_drive/mass_train_ROI_crop/CBIS-DDSM/Mass-Training_P_00044_RIGHT_MLO_2/07-21-2016-DDSM-98278/1-ROI mask images-14207/000001.dcm')
# Mass-Training_P_00044_RIGHT_CC_2
# Mass-Training_P_00044_RIGHT_CC_3
# Mass-Training_P_00044_RIGHT_CC_4
# Mass-Training_P_00044_RIGHT_MLO_1
# Mass-Training_P_00044_RIGHT_MLO_2

# calc_full = pydicom.read_file(
# calc_roi = pydicom.read_file('/Volumes/chris_external_drive/calc_train_ROI_crop/CBIS-DDSM/Calc-Training_P_00005_RIGHT_CC_1/08-30-2017-DDSM-09081/1-cropped images-94682/000000.dcm'
# calc_crop = pydicom.read_file('/Volumes/chris_external_drive/calc_train_ROI_crop/CBIS-DDSM/Calc-Training_P_00005_RIGHT_CC_1/08-30-2017-DDSM-09081/1-cropped images-94682/000001.dcm'

# fig, axs = plt.subplots(2,2)
# axs[0, 0].imshow(ds_cc.pixel_array)
# axs[0, 0].set_title('Craniocaudal view')
# axs[0, 1].imshow(ds2_mlo.pixel_array)
# axs[0, 1].set_title('Mediolateral Oblique view')
# axs[1, 0].imshow(ds_roi_1.pixel_array)
# axs[1, 0].set_title('Region of interest marked')
# axs[1, 1].imshow(ds_roi_2.pixel_array)
# axs[1, 1].set_title('Region of interest cropped')
# plt.tight_layout()
# plt.savefig('/Users/christopherlawton/galvanize/module_2/capstone_2/proposal/full_cc_mlo_roi.png')


#mass dataframes
# mass_train = pd.read_csv('./capstone_2/proposal/mass_case_description_train_set (6).csv')
# mass_test = pd.read_csv('./capstone_2/proposal/mass_case_description_test_set (2).csv')
# mass_full = pd.concat([mass_train, mass_test])
# mass_cc = mass_full[mass_full['image view'] == 'CC']
#
# #calc dataframes
# calc_train = pd.read_csv('./capstone_2/proposal/calc_case_description_train_set (2).csv')
# calc_test = pd.read_csv('./capstone_2/proposal/calc_case_description_test_set.csv')
# calc_full = pd.concat([calc_train, calc_test])
# calc_cc = calc_full[calc_full['image view'] == 'CC']
