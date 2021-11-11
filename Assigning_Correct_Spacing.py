__author__ = 'Brian M Anderson'
# Created on 6/3/2020

import SimpleITK as sitk
import os
from Resample_Class.Resample_Class import Resample_Class_Object


base_path = r'H:\Liver_Disease_Ablation'

for file_path in files:
    image_path = os.path.join(base_path,file_path)
    iteration = os.path.split(image_path)[-1].split('_')[-1].split('.')[0]
    annotation_path = image_path.replace('Overall_Data','Overall_mask').replace('_{}'.format(iteration),'_y{}'.format(iteration))
    if os.path.exists(image_path):
        os.rename(image_path,os.path.join(new_path,os.path.split(image_path)[-1]))
    if os.path.exists(annotation_path):
        os.rename(annotation_path, os.path.join(new_path, os.path.split(annotation_path)[-1]))
    # image_handle = sitk.ReadImage(image_path)
    # annotation_handle = sitk.ReadImage(annotation_path)
    # new_spacing = (0.9, 0.9, 2.0)
    # image_handle.SetSpacing(new_spacing)
    # annotation_handle.SetSpacing(new_spacing)
    # sitk.WriteImage(image_handle, image_path)
    # sitk.WriteImage(annotation_handle, annotation_path)
    # xxx = 1