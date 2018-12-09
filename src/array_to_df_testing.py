import os,sys
import shutil
import pydicom
import pandas as pd
import numpy as np

root_directory ='/Volumes/chris_external_drive/Mass-train-CBIS-DDSM'

count = 0
full = []
for dir,subdirs,files in sorted(os.walk(root_directory)):
        for file in files:
            if count < 5:
                fullpath = os.path.realpath( os.path.join(dir,file) )
                dirname = fullpath.split(os.path.sep)
                ds = pydicom.read_file(fullpath)
                ls = []
                ls.append(dirname[4][14:21]) #naming specific to training sets
                ls.append(dirname[4][22])
                ls.jhappend(ds.PatientOrientation)
                ls.append([ds.pixel_array])
                ls.append(ds.pixel_array.dtype)
                full.append(ls)
                count+=1
                print(count) #just to confirm it's running
            else:
                break

df = pd.DataFrame(full, columns=['patient_id', 'laterality', 'orientation', 'image', 'dtype'])
df.to_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/test_arr.csv')
