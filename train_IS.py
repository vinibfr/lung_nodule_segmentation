import pandas as pd
import argparse
import os
import numpy as np
from collections import OrderedDict
import yaml
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from losses import BCEDiceLoss
from datasetIS import MyLidcDataset
from metrics import iou_score,dice_coef, precision, recall
from utils import AverageMeter, str2bool
import itertools
import time
import psutil
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# models basic
from models.UNet import UNet
from models.NestedUnet import NestedUNet
from models.AttentionUNet import AttentionUNet
from models.UTNet import UTNet

#models from skip
from models.SAMAttentionUNet import SAMAttentionUNet
from models.MixAttentionUNet import MixAttentionUNet
from models.MixAttentionUNetClicks import MixAttentionUNetClicks
from models.MixAttentionUNetClicksV2 import MixAttentionUNetClicksV2
from models.SAMNestedUnet import SAMNestedUnet
from models.R2AttentionUNet import R2AttentionUNet
from models.R2AttentionUNetSAM import R2AttentionUNetSAM
from models.SKSAMUNet import SKSAMUNet
from models.SKSAMUNetV2 import SKSAMUNetV2

#models from bottleneck
from models.ASPPUNet import ASPPUnet
from models.PYBUNet import PYBUNet
from models.SEBUNet import SEBUNet
from models.SAMUNet import SAMUNet
from models.SAMUNet2 import SAMUNet2
from models.VITUNet import VITUNet

def log_time():
    return time.time()

def to_MB(a):
    return a/1024.0/1024.0

def get_gpu_memory_usage():
    return round(torch.cuda.memory_allocated() / 1024**3), round(torch.cuda.max_memory_allocated() / 1024**3)

def get_ram_usage():
    return psutil.virtual_memory().used / (1024**3)

def get_comma_separated_int_args(value):
    value_list = value.split(',')
    return [int(i) for i in value_list]

def return_model(config):
    # basic models
    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=2, input_channels=2)
    elif config['name']=='VITUNet':
        model = VITUNet(n_channels=2, n_classes=2, bilinear=True)
    elif config['name']=='UTNet':
        model = UTNet(2, config['base_chan'], config['num_class'], reduce_size=config['reduce_size'], block_list=config['block_list'], num_blocks=config['num_blocks'], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=config['aux_loss'], maxpool=True)        
    elif config['name']=='AttentionUNet':
        model = AttentionUNet(n_classes=1, in_channel=2, out_channel=2)
    elif config['name']=='ASPPUNet': # bottleneck models
        model = ASPPUnet(2,2)
    elif config['name']=='PYBUNet':
        model = PYBUNet(2,2)
    elif config['name']=='SEBUNet':
        model = SEBUNet(2,2)
    elif config['name']=='SAMUNet':
        model = SAMUNet(2,2)
    elif config['name']=='SAMUNet2':
        model = SAMUNet2(2,2)        
    elif config['name']=='SAMNestedUnet': # skip models
        model = SAMNestedUnet(num_classes=2, input_channels=2)
    elif config['name']=='MixAttentionUNet':
        model = MixAttentionUNet(n_classes=1, in_channel=2, out_channel=2)
    elif config['name']=='MixAttentionUNetClicks':
        model = MixAttentionUNetClicks(n_classes=1, in_channel=1, out_channel=1)        
    elif config['name']=='MixAttentionUNetClicksV2':
        model = MixAttentionUNetClicksV2(n_classes=1, in_channel=1, out_channel=1)
    elif config['name']=='R2AttentionUNet':
        model = R2AttentionUNet(2,2)
    elif config['name']=='R2AttentionUnetSAM':
        model = R2AttentionUNetSAM(2,2)
    elif config['name']=='SKSAMUNet':
        model = SKSAMUNet(2,2)
    elif config['name']=='SKSAMUNetV2':
        model = SKSAMUNetV2(2,2)
    elif config['name']=='SAMAttentionUNet':
        model = SAMAttentionUNet(n_classes=1, in_channel=2, out_channel=2)        
    else:
        model = UNet(2,2) # basic model unet
    return model, config

def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--name', default="SAMUNet",
                        help='model names')
    parser.add_argument('--epochs', default=85, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=60, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=14, type=int)
    parser.add_argument('--channels', default=1, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--smr', '--smooth_rate_loss', default=1e-5, type=float,
                        metavar='smr', help='smooth rate for loss')    
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--tensorboard', default=True, type=str2bool,
                        help='use tensorboard for visualization')
    # data
    parser.add_argument('--augmentation',type=str2bool,default=True,choices=[True,False])


    config = parser.parse_args()

    return config


def train(train_loader, model, criterion, smooth, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'prec': AverageMeter(),
                  'rec': AverageMeter()}

    model.train()

    if os.name == 'nt':
        pbar = tqdm(total=len(train_loader))

    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target, smooth)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)
        prec = precision(output, target)
        rec = recall(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['prec'].update(prec, input.size(0))
        avg_meters['rec'].update(rec, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg),
            ('prec',avg_meters['prec'].avg),
            ('rec',avg_meters['rec'].avg)
        ])
        if os.name == 'nt':
            pbar.set_postfix(postfix)
            pbar.update(1)
    if os.name == 'nt':
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg),
                        ('prec',avg_meters['prec'].avg),
                        ('rec',avg_meters['rec'].avg)])

def validate(val_loader, model, criterion, smooth):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'prec': AverageMeter(),
                  'rec': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    if os.name == 'nt':
        pbar = tqdm(total=len(val_loader))
    with torch.no_grad():
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()
            np.save('notebook/input_val.npy',input.cpu().numpy())
            np.save('notebook/target_val.npy',target.cpu().numpy())
            output = model(input)
            loss = criterion(output, target, smooth)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            prec = precision(output, target)
            rec = recall(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['prec'].update(prec, input.size(0))
            avg_meters['rec'].update(rec, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg),
                ('prec',avg_meters['prec'].avg),
                ('rec',avg_meters['rec'].avg)
            ])
        if os.name == 'nt':
            pbar.set_postfix(postfix)
            pbar.update(1)
    if os.name == 'nt':
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg),
                        ('prec',avg_meters['prec'].avg),
                        ('rec',avg_meters['rec'].avg)
                        ])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main_hyper():
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")    
    print("main started:",formatted_date)
    # Get configuration
    config = vars(parse_args())
    # Make Model output directory
    
    # set hyperparameters possible values
    hyper_model = ['UNet','NestedUNET','AttentionUNet','SAMUNet', 'MixAttentionUNet','MixAttentionUNetClicks','ASPPUNet','PYBUNet', 'SEBUNet', 'SAMUNet2','VITUNet','SAMNestedUnet','SKSAMUNet','SKSAMUNetV2','R2AttentionUNet','R2AttentionUnetSAM','MixAttentionUNetClicks']
    hyper_model = ['MixAttentionUNetClicks']
    hyper_augmentation = [False]
    hyper_lr = [1e-4]
    if os.name == 'nt':
        hyper_batch_size = [10]
    else:
        hyper_batch_size = [10]
    hyper_loss_smooth = [1e-4]
    hyper_crop_size = [512]
    hyper_meta = ['meta']
    sep = False
    # Generate all combinations of hyperparameters
    hyperparameter_combinations = itertools.product(
        hyper_batch_size, hyper_model, hyper_augmentation, hyper_lr, hyper_loss_smooth, hyper_crop_size, hyper_meta
    )
    for batch_size, model_test, augmentation, learning_rate, loss_smooth, crop_size, metas in hyperparameter_combinations:         
        config['augmentation'] = augmentation
        config['lr'] = learning_rate
        if model_test == 'R2AttentionUNet' or model_test == 'R2AttentionUnetSAM':
            batch_size = 10 
        config['batch_size'] = batch_size
        config['name'] = model_test
        config['smr'] = loss_smooth
        config['channels'] = 2

        if model_test == 'MixAttentionUNetClicksV2' or model_test == 'MixAttentionUNetClicks':
            sep = True
            config['batch_size'] = 10   

        file_name = config['name'] + 'interactive_segmentation__false_batch_size_{}_epochs_{}_crop_{}_{}'.format(batch_size, config['epochs'], str(crop_size),metas)
        os.makedirs('models_output_DA_{}/{}'.format(str(crop_size),file_name),exist_ok=True)
        print("Creating directory called",file_name)

        print('-' * 20)
        print("Configuration Setting as follow")
        for key in config:
            print('{}: {}'.format(key, config[key]))
        print('-' * 20)

        #save configuration
        with open('models_output_DA_{}/{}/config.yml'.format(str(crop_size),file_name), 'w') as f:
            yaml.dump(config, f)

        #criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = BCEDiceLoss().cuda()
        cudnn.benchmark = True

        # create model
        model, config = return_model(config)
        model = model.cuda()
        total_params = count_parameters(model)
        print("Total parameters:", total_params)        
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        else:
            print("Let's use", torch.cuda.device_count(), "CPUs!")
        print(f"Memory after model to device: {to_MB(torch.cuda.memory_allocated()):.2f}MB")
        params = filter(lambda p: p.requires_grad, model.parameters())

        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        
        if os.name == 'nt':
            print("Running on Windows")
            IMAGE_DIR = 'D:/data/Test/ImageCL' + str(crop_size)
            MASK_DIR = 'D:/data/Test/MaskCL' + str(crop_size)
        else:
            print("Running on Linux")
            IMAGE_DIR = '/nas-ctm01/datasets/private/LUCAS/LIDC-IS/ImageCL512'
            MASK_DIR = '/nas-ctm01/datasets/public/LIDC/MaskCL512'

        meta = pd.read_csv(metas + '.csv')                  
        ############################################################################
        # Get train/test label from meta.csv
        meta['original_image'] = meta['original_image'].apply(lambda x: IMAGE_DIR + '/' + x + '.npy')
        meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ '/' + x +'.npy')

        train_meta = meta[meta['data_split']=='Train']
        val_meta = meta[meta['data_split']=='Validation']

        # Get all *npy images into list for Train
        train_image_paths = list(train_meta['original_image'])
        train_mask_paths = list(train_meta['mask_image'])

        # Get all *npy images into list for Validation
        val_image_paths = list(val_meta['original_image'])
        val_mask_paths = list(val_meta['mask_image'])
        print("*"*50)
        print("The lenght of image: {}, mask folders: {} for train".format(len(train_image_paths),len(train_mask_paths)))
        print("The lenght of image: {}, mask folders: {} for validation".format(len(val_image_paths),len(val_mask_paths)))
        print("Ratio between Val/ Train is {:2f}".format(len(val_image_paths)/len(train_image_paths)))
        print("*"*50)

        # Create Dataset
        train_dataset = MyLidcDataset(train_image_paths, train_mask_paths,crop_size,config['augmentation'],sep)
        val_dataset = MyLidcDataset(val_image_paths,val_mask_paths,crop_size,False,sep)

        # Create Dataloader
        print("Creating Dataloader Train")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=config['num_workers'])
        print("Creating Dataloader Validation")
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=config['num_workers'])
        log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

        best_dice = 0
        trigger = 0

        print("Start Training")
        for epoch in range(config['epochs']):

            # train for one epoch
            start_time = log_time()
            train_log = train(train_loader, model, criterion, config['smr'],optimizer)
            end_time = log_time()
            elapsed_train_time = int(round(end_time - start_time))
            vram_current_training, vram_peak_training = get_gpu_memory_usage()
            ram_used_training = round(get_ram_usage())

            # evaluate on validation set
            start_time = log_time()
            val_log = validate(val_loader, model, criterion,config['smr'])
            end_time = log_time()
            elapsed_val_time = int(round(end_time - start_time))
            vram_current_val, vram_peak_val = get_gpu_memory_usage()
            ram_used_val = round(get_ram_usage())
            time.sleep(20)
            print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
                epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'],val_log['iou']))

            tmp = pd.Series(
                [epoch,
                config['lr'],
                train_log['loss'],
                train_log['iou'],
                train_log['dice'],
                val_log['loss'],
                val_log['iou'],
                val_log['dice'],
                elapsed_train_time,
                elapsed_val_time,
                vram_current_training,
                vram_current_val,
                vram_peak_training,
                vram_peak_val,
                ram_used_training,
                ram_used_val],
                index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice', 'elapsed_train_time', 'elapsed_val_time', 'vram_current_training', 'vram_current_val', 'vram_peak_training', 'vram_peak_val', 'ram_used_training', 'ram_used_val'])
            log = log._append(tmp, ignore_index=True)
            # Save the updated log DataFrame to CSV
            log.to_csv('models_output_DA_{}/{}/log.csv'.format(str(crop_size),file_name), index=False)
            trigger += 1
            if val_log['dice'] > best_dice:
                torch.save(model.state_dict(), 'models_output_DA_{}/{}/model.pth'.format(str(crop_size),file_name))
                best_dice = val_log['dice']
                print("=> saved best model as validation DICE is greater than previous best DICE: {:.4f} of epoch:{} from file:{}".format(best_dice,epoch,file_name))
                trigger = 0

            # early stopping
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> early stopping")
                break

            torch.cuda.empty_cache()
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")    
        print("main ended:",formatted_date)


if __name__ == '__main__':
    main_hyper()