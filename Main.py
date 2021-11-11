__author__ = 'Brian M Anderson'
# Created on 9/8/2020

'''
Obtain image parameters
'''
obtain_parameters = False
if obtain_parameters:
    from Obtain_Image_Acquisition_Parameters import obtain_image_parameters
    obtain_image_parameters()

'''
First, convert the nii files to dicom
'''
convert_lits_to_dicom = False
if convert_lits_to_dicom:
    from convert_lits_nii_to_dicom import *
    convert_nii_to_dicom()

'''
Upload the dicom files to Raystation for viewing and liver/disease prediction
'''


'''
Next, create predictions for the liver from the dicom files
'''

#  Run Create_Liver_Contours
'''
Edit the liver contours as needed, ensure that potential disease is included as 'liver'
'''

'''
Some files do not have the correct orientation...
'''
correct_orientation = False
if correct_orientation:
    from Assign_correct_orientation import assign_orientation
    assign_orientation()
image_cube = (5, 32, 32, 4) # This is the image cube I will use for training


My_amazing_unzip_and_worker_thing