import pandas as pd
import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np
import itertools
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# v2 for 2 channels
#from datasetV2 import MyLidcDataset

# test for 1 channel
from dataset import MyLidcDataset
from metrics import iou_score,dice_coef,dice_coef2
from utils import AverageMeter, str2bool

# models
from models.MixAttentionUNet import MixAttentionUNet
from models.SAMUNet import SAMUNet
from models.SAMUNet2 import SAMUNet2
from models.UNet import UNet
from models.NestedUnet import NestedUNet
from models.AttentionUNet import AttentionUNet
from models.VITUNet import VITUNet

def return_model(config, device):
    if config['name']=='NestedUNet':
        model = NestedUNet(num_classes=1, input_channels=1).to(device)
    elif config['name']=='MixAttentionUNet':
        model = MixAttentionUNet(n_classes=1, in_channel=1, out_channel=1).to(device)        
    elif config['name']=='AttentionUNet':
        model = AttentionUNet(n_classes=1, in_channel=1, out_channel=1).to(device)
    elif config['name']=='SAMUNet':
        model = SAMUNet(1, 1).to(device)
    elif config['name']=='SAMUNet2':
        model = SAMUNet2(1, 1).to(device)        
    elif config['name']=='VITUNet':
        model = VITUNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    else:
        model = UNet(2,2).to(device)
    return model, config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="SAMUNet",
                        help='model names')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--augmentation',default=False,type=str2bool,
                help='Shoud we get the augmented version?')
    parser.add_argument('--num_workers', default=14, type=int)
    args = parser.parse_args()

    return args

def save_output(output,output_directory,test_image_paths,counter):
    # This saves the predicted image into a directory. The naming convention will follow PI
    for i in range(output.shape[0]):
        label = test_image_paths[counter][-23:]
        label = label.replace('NI','PD')
        np.save(output_directory+'/'+label,output[i,:,:])
        #print("SAVED",output_directory+label+'.npy')
        counter+=1


    return counter

def calculate_fp(prediction_dir,mask_dir,distance_threshold=80):
    """This calculates the fp by comparing the predicted mask and orginal mask"""
    #TP,TN,FP,FN
    #FN will always be zero here as all the mask contains a nodule
    confusion_matrix =[0,0,0,0]
    # This binary structure enables the function to recognize diagnoally connected label as same nodule.
    # Load a sample prediction to determine the structure dimensions
    sample_prediction_file = os.listdir(prediction_dir)[0]
    sample_predict = np.load(os.path.join(prediction_dir, sample_prediction_file))
    dimensions = sample_predict.ndim

    # Create the structure array based on the dimensions of predict
    s = generate_binary_structure(dimensions, dimensions)
    print('Length of prediction dir is ',len(os.listdir(prediction_dir)))
    for prediction in os.listdir(prediction_dir):
        #print(confusion_matrix)
        mask_id = prediction.replace('PD','MA')
        mask = np.load(mask_dir+'/'+mask_id)
        predict = np.load(prediction_dir+'/'+prediction)
        answer_com = np.array(ndi.center_of_mass(mask))
        # Patience is used to check if the patch has cropped the same image
        patience =0
        labeled_array, nf = label(predict, structure=s)
        if nf>0:
            for n in range(nf):
                lab=np.array(labeled_array)
                lab[lab!=(n+1)]=0
                lab[lab==(n+1)]=1
                predict_com=np.array(ndi.center_of_mass(labeled_array))
                if np.linalg.norm(predict_com-answer_com,2) < distance_threshold:
                    patience +=1
                else:
                    confusion_matrix[2]+=1
            if patience > 0:
                # Add to True Positive
                confusion_matrix[0]+=1
            else:
                # Add to False Negative
                # if the patience remains 0, and nf >0, it means that the slice contains both the TN and FP
                confusion_matrix[3]+=1

        else:
            # Add False Negative since the UNET didn't detect a cancer even when there was one
            confusion_matrix[3]+=1
    return np.array(confusion_matrix)

def calculate_fp_clean_dataset(prediction_dir,distance_threshold=80):
    """This calculates the confusion matrix for clean dataset"""
    #TP,TN,FP,FN
    #When we calculate the confusion matrix for clean dataset, we can only get TP and FP.
    # TP - There is no nodule, and the segmentation model predicted there is no nodule
    # FP - There is no nodule, but the segmentation model predicted there is a nodule
    confusion_matrix =[0,0,0,0]
    s = generate_binary_structure(2,2)
    for prediction in os.listdir(prediction_dir):
        predict = np.load(prediction_dir+'/'+prediction)
        # Patience is used to check if the patch has cropped the same image
        patience =0
        labeled_array, nf = label(predict, structure=s)
        if nf>0:
            previous_com = np.array([-1,-1])
            for n in range(nf):
                lab=np.array(labeled_array)
                lab[lab!=(n+1)]=0
                lab[lab==(n+1)]=1
                predict_com=np.array(ndi.center_of_mass(labeled_array))
                if previous_com[0] == -1:
                    # add to false positive
                    confusion_matrix[2]+=1
                    previous_com = predict_com
                    continue
                else:
                    if np.linalg.norm(previous_com-predict_com,2) > distance_threshold:
                        if patience != 0:
                            #print("This nodule has already been taken into account")
                            continue
                        # add false positive
                        confusion_matrix[2]+=1
                        patience +=1

        else:
            # Add True Negative since the UNET didn't detect a cancer even when there was one
            confusion_matrix[1]+=1

    return np.array(confusion_matrix)

def main():
    config = vars(parse_args())
    
    hyper_model_dict = [
    {"model": "UNet", "file": "UNetfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    {"model": "UNet", "file": "UNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},     
     #{"model": "UNet", "file": "UNetfull_initMETA_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
     #{"model": "VITUNet", "file": "VITUNetfull_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
     #{"model": "SAMUNet", "file": "SAMUNetfull_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
     #{"model": "AttentionUNet", "file": "AttentionUNetfull_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
     #{"model": "MixAttentionUNet", "file": "MixAttentionUNetfull_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
     #{"model": "NestedUNet", "file": "NestedUNETfull_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},     
     {"model": "UNet", "file": "Unetfull_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
     #{"model": "SAMUNet2", "file": "SAMUNet2full_skip_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
    #{"model": "AttentionUNet", "file": "AttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "AttentionUNet", "file": "AttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "MixAttentionUNet", "file": "MixAttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "MixAttentionUNet", "file": "MixAttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "NestedUNET", "file": "NestedUNETfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "NestedUNET", "file": "NestedUNETfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "SAMUNet", "file": "SAMUNetfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "SAMUNet", "file": "SAMUNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "NestedUNet", "file": "NestedUNETfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "AttentionUNet", "file": "AttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "UNet", "file": "UNetfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "AttentionUNet", "file": "AttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "UNet", "file": "UNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "MixAttentionUNet", "file": "MixAttentionUNetfull_transfer_batch_size_20_epochs_80_crop_512_augTrue_optmi_loaded_points_fivepoint","crop_size":512},
    #{"model": "MixAttentionUNet", "file": "MixAttentionUNetinteractive_segmentation_fivepoint_False_test_batch_size_20_epochs_85_crop_512","crop_size":512},
    #{"model": "SAMUNet", "file": "SAMUNetinteractive_segmentation_fivepoint_False_test_batch_size_20_epochs_85_crop_512","crop_size":512},
    #{"model": "NestedUNET", "file": "NestedUNETfull_transfer_batch_size_20_epochs_80_crop_512_augFalse_optmi_loaded_points_fivepoint","crop_size":512},
    {"model": "UNet", "file": "UNetfull_initMETA_batch_size_20_epochs_80_crop_512_augTrue","crop_size":512},
    ]
    log= pd.DataFrame(index=[],columns= ['model','iou','dice'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for item  in hyper_model_dict:
        print("="*50)    
        model_name = item['model']
        crop_size = item['crop_size']
        print("Model sequence is ",model_name)
        config['name'] = model_name
        NAME = item['file']
        # load configuration
        with open('models_output_{}/{}/config.yml'.format(str(crop_size),NAME), 'r') as f:
            model_config = yaml.safe_load(f)

        print('-'*20)
        for key in model_config.keys():
            print('%s: %s' % (key, str(model_config[key])))
        print('-'*20)

        cudnn.benchmark = True

        # create model
        print("=> creating model {}".format(NAME))
        model, config = return_model(config, device)
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        print("Loading model file from {}".format(NAME))
        model.load_state_dict(torch.load('models_output_{}/{}/model.pth'.format(str(crop_size),NAME),map_location=device))

        if os.name == 'nt':
            print("Running on Windows")
            if str(crop_size) == '512':
                IMAGE_DIR = 'D:/data/Test/ImageCL512/'
                MASK_DIR = 'D:/data/Test/MaskCL512/'
            else:
                IMAGE_DIR = 'D:/data/Image' + str(crop_size) + '/'
                MASK_DIR = 'D:/data/Mask' + str(crop_size) + '/'
        else:
            print("Running on Linux")
            IMAGE_DIR = '/nas-ctm01/datasets/private/LUCAS/LIDC-Full/data/Image/'
            MASK_DIR = '/nas-ctm01/datasets/private/LUCAS/LIDC-Full/data/Mask/'
        IMAGE_DIR = 'F:/mestrado/framework_lung_segmentation/preprocessing/data_new/image/'
        MASK_DIR = 'F:/mestrado/framework_lung_segmentation/preprocessing/data_new/masks/'

        #IMAGE_DIR = 'D:/data/Image512/'
        #MASK_DIR = 'D:/data/Mask512/'

        meta = pd.read_csv('meta.csv')   

        # Get train/test label from meta.csv
        meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
        meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')
        test_meta = meta[meta['data_split']=='Test']

        # Get all *npy images into list for Test(True Positive Set)
        test_image_paths = list(test_meta['original_image'])
        test_mask_paths = list(test_meta['mask_image'])

        total_patients = len(test_meta.groupby('patient_id'))

        print("*"*50)
        print("The lenght of image: {}, mask folders: {} for test".format(len(test_image_paths),len(test_mask_paths)))
        print("Total patient number is :{}".format(total_patients))


        # Directory to save U-Net predict output
        OUTPUT_MASK_DIR = './Segmentation_output_eval_nofilter_{}/{}'.format(str(crop_size),NAME)
        print("Saving OUTPUT files in directory {}".format(OUTPUT_MASK_DIR))
        os.makedirs(OUTPUT_MASK_DIR,exist_ok=True)


        test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=model_config['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=model_config['num_workers'])
        model.eval()
        print(" ")
        avg_meters = {'iou': AverageMeter(),
                    'dice': AverageMeter()}
        
        with torch.no_grad():

            counter = 0
            pbar = tqdm(total=len(test_loader))
            for input, target in test_loader:

                # add a channel to the image
                input = input.cuda()
                target = target.cuda()

                output = model(input)
                iou = iou_score(output, target)
                dice = dice_coef2(output, target)

                avg_meters['iou'].update(iou, input.size(0))
                avg_meters['dice'].update(dice, input.size(0))

                postfix = OrderedDict([
                    ('iou', avg_meters['iou'].avg),
                    ('dice',avg_meters['dice'].avg)
                ])
                output = torch.sigmoid(output)
                output = (output>0.5).float().cpu().numpy()

                counter = save_output(output,OUTPUT_MASK_DIR,test_image_paths,counter)
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

            tmp = pd.Series(
                [NAME,
                avg_meters['iou'].avg,
                avg_meters['dice'].avg],
                index=['model', 'iou', 'dice'])
            log = log._append(tmp, ignore_index=True)
            log.to_csv('Segmentation_output_eval_nofilter_{}/{}/validation_log_filter.csv'.format(str(crop_size),NAME), index=False)
        print("="*50)
        print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
        print('DICE:{:.4f}'.format(avg_meters['dice'].avg))

        '''Calculate the confusion matrix for the clean dataset
        confusion_matrix = calculate_fp(OUTPUT_MASK_DIR ,MASK_DIR,distance_threshold=80)
        print("="*50)
        print("TP: {} FP:{}".format(confusion_matrix[0],confusion_matrix[2]))
        print("FN: {} TN:{}".format(confusion_matrix[3],confusion_matrix[1]))
        print("{:2f} FP/per Scan ".format(confusion_matrix[2]/total_patients))
        print("="*50)
        print(" ")
        torch.cuda.empty_cache()'''

if __name__ == '__main__':
    main()