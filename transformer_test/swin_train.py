import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from losses import BCEDiceLoss
from dataset import MyLidcDataset
from metrics import iou_score,dice_coef, precision, recall
from utils import AverageMeter, str2bool
import itertools
# models
from swin_unet.vision_transformer import SwinUnet as ViT_seg
from swin_unetv2.vision_transformer import SwinUnet as ViT_segV2
from swin_unet.config import get_config

from inputimeout import inputimeout 

def get_input_with_timeout(prompt, timeout=10, default_value=None):
    try: 
        time_over = inputimeout(prompt=prompt, timeout=timeout) 
    except Exception: 
        time_over = default_value
    return time_over

def get_comma_separated_int_args(value):
    value_list = value.split(',')
    return [int(i) for i in value_list]

def return_model(config, args):
    if config['name'] == 'SwinUNET':
        model =  ViT_seg(args, img_size=config['img_size'], num_classes=config['num_classes']).cuda()
    elif config['name'] == 'SwinUNETV2':
        model =  ViT_segV2(args, img_size=config['img_size'], num_classes=config['num_classes']).cuda()
    if config['preload']:
        model.load_from(args)
    return model, config

def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--early_stopping', default=60, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=14, type=int)
    parser.add_argument('--name', default="UNet",
                        help='model names')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')   
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')    
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')                   
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--epochs', type=int,
                        default=80, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.0001,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--augmentation',type=str2bool,default=True,choices=[True,False])
    parser.add_argument('--preload',type=str2bool,default=True,choices=[True,False])
    parser.add_argument('--cfg', type=str, default='swin_unet/configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--tensorboard', default=True, type=str2bool,
                        help='use tensorboard for visualization')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')    
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
    args = get_config(config)
    # Make Model output directory
    
    # set hyperparameters possible values
    hyper_model = ['SwinUNET','SwinUNETV2']
    hyper_augmentation = [True]
    hyper_lr = [1e-4]
    if os.name == 'nt':
        hyper_batch_size = [24]
    else:
        hyper_batch_size = [24]

    hyper_loss_smooth = [1e-6]
    hyper_preload = [True,False]
    hyper_weight_decay = [1e-4]
    hyper_betas = [(0.9, 0.999)]
    hyper_crop_size = [224]
    if os.name == 'nt':
        hyper_meta = ['meta_filtered_full']
    else:
        hyper_meta = ['meta']

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = itertools.product(
        hyper_model, hyper_augmentation, hyper_lr, hyper_loss_smooth, hyper_weight_decay, hyper_betas, hyper_crop_size,  hyper_meta, hyper_preload, hyper_batch_size
    )    

    for model_test, augmentation, learning_rate, loss_smooth, weight_decay, betas, crop_size, metas, load_state,batch_size in hyperparameter_combinations:               
        config['augmentation'] = augmentation
        config['base_lr'] = learning_rate
        config['batch_size'] = batch_size
        config['name'] = model_test
        config['smr'] = loss_smooth
        config['weight_decay'] = weight_decay
        config['preload'] = load_state
        file_name = config['name'] + 'full_augmentation_{}_lr_{}_batch_size_{}_epochs_{}_crop_{}_metafile_{}_preload_{}'.format(augmentation, learning_rate, batch_size, config['epochs'], str(crop_size), metas,load_state)
        os.makedirs('./models_output/{}'.format(file_name),exist_ok=True)    
        print("Creating directory called",file_name)


        print('-' * 20)
        print("Configuration Setting as follow")
        for key in config:
            print('{}: {}'.format(key, config[key]))
        print('-' * 20)

        #save configuration
        with open('./models_output/{}/config.yml'.format(file_name), 'w') as f:
            yaml.dump(config, f)

        #criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = BCEDiceLoss().cuda()
        cudnn.benchmark = True

        # create model
        model, config = return_model(config, args)
        model = model.cuda()
        total_params = count_parameters(model)
        print("Total parameters:", total_params)        
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        else:
            print("Let's use", torch.cuda.device_count(), "CPUs!")

        params = filter(lambda p: p.requires_grad, model.parameters())

        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(params, lr=config['base_lr'], weight_decay=config['weight_decay'], betas=betas)
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=config['base_lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        if os.name == 'nt':
            print("Running on Windows")
            IMAGE_DIR = 'F:/data/lidc/original/Image' + str(crop_size)
            MASK_DIR = 'F:/data/lidc/original/Mask' + str(crop_size)
        else:
            print("Running on Linux")
            IMAGE_DIR = '/nas-ctm01/homes/vbreis/data/Image' + str(crop_size)
            MASK_DIR = '/nas-ctm01/homes/vbreis/data/Mask' + str(crop_size)
        print("Image directory:",IMAGE_DIR)
        print("Mask directory:",MASK_DIR)
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
        train_dataset = MyLidcDataset(train_image_paths, train_mask_paths,crop_size,config['augmentation'])
        # add model graph to tensorboard
        val_dataset = MyLidcDataset(val_image_paths,val_mask_paths,crop_size,False)

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
        # Visualize model in TensorBoard
        log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

        best_dice = 0
        trigger = 0

        print("Start Training")
        for epoch in range(config['epochs']):

            # train for one epoch
            print("Training")
            train_log = train(train_loader, model, criterion, config['smr'],optimizer)

            # evaluate on validation set
            val_log = validate(val_loader, model, criterion,config['smr'],)

            print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
                epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'],val_log['iou']))

            tmp = pd.Series(
                [epoch,
                config['base_lr'],
                train_log['loss'],
                train_log['iou'],
                train_log['dice'],
                val_log['loss'],
                val_log['iou'],
                val_log['dice']],
                index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])
            log = log._append(tmp, ignore_index=True)
            # Save the updated log DataFrame to CSV
            log.to_csv('models_output/{}/log.csv'.format(file_name), index=False)
            trigger += 1
            if val_log['dice'] > best_dice:
                torch.save(model.state_dict(), 'models_output/{}/model.pth'.format(file_name))
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