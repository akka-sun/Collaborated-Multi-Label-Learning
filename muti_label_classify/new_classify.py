import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

from tqdm import tqdm

from models.resnet101 import ResNet101_Multiclass
from models.clip_muti import CLIP, CLIP_mld
from models.ml_decoder import add_ml_decoder_head
from tresnet.tresnet import TResnetXL
from dataset.get_new_dataset import get_datasets
from models.loss import MyLoss, TwoWayLoss, AsymmetricLoss, BCELoss
from models.pasl import PartialSelectiveLoss
from models.splc import SPLC
from models.ll import compute_batch_loss
from utils.logger import Logger
from utils.metrics import voc_mAP,get_PR
from utils.earlystop import EarlyStopping

batch_size = 256
img_size=512
epochs=300
stop_epoch = 1000
print_freq=200

data_name='food_gemma'
loss_name='ll'
best_mAP=0
best_of1=0
if 'coco' in data_name:
    num_classes = 80
    train_img_dir = 'your_root_to_data'
    val_img_dir = 'your_root_to_data'
elif 'ade20k' in data_name:
    num_classes = 150
    train_img_dir = 'your_root_to_data'
    val_img_dir = 'your_root_to_data'
elif 'food' in data_name:
    num_classes = 102
    train_img_dir = 'your_root_to_data'
    val_img_dir = 'your_root_to_data'
data_type='modify'
model_name='clip'
train_json_dir=f'/root/{data_name}_json/{data_name}_{data_type}.json'
val_json_dir=f'/root/{data_name}_json/{data_name}_val.json'
train_ai_json_dir = f'/root/{data_name}_json/{data_name}_ai.json'
val_ai_json_dir = f'/root/{data_name}_json/{data_name}_val.json'

lr=1e-4
weight_decay=1e-4

output='your_root_to_output'
resume=''

if not os.path.exists(output):
    os.makedirs(output)


def main():
    global best_mAP
    global best_of1
    if loss_name=='mld':
        model = CLIP_mld(num_classes)
        model = add_ml_decoder_head(model, num_classes=num_classes, num_of_groups=-1,
                                    decoder_embedding=768, zsl=0)
    else:
        model = CLIP(num_classes)
    model = model.cuda()
    train_dataset, val_dataset = get_datasets(data_name,train_img_dir,train_json_dir,train_ai_json_dir,val_img_dir,val_json_dir,val_ai_json_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = MyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stop = EarlyStopping(patience=stop_epoch)


    logger=Logger(f"{data_name}-{data_type}",log_dir=output).logger
    start_epoch=0
    total_time = 0
    if resume:
        if os.path.isfile(resume):
            logger.info("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            del checkpoint
            torch.cuda.empty_cache()

    for epoch in range(start_epoch,epochs):
        train(train_loader, model, criterion, optimizer, epoch, logger)
        mAP = 0
        OF1 = 0
        OR = 0
        OP = 0
        if (epoch+1)%1 == 0:
            mAP,OF1,OP,OR=validate(val_loader,model,logger)
        if early_stop(mAP,epoch):
            logger.info(f'best mAP:{best_mAP}')
            break
        if mAP>best_mAP and OF1*100>best_of1:
            best_of1 = OF1*100
            best_op = OP
            best_or = OR
            best_mAP = max(mAP, best_mAP)
            state_dict = model.state_dict()
            torch.save({'state_dict': state_dict,'optimizer_state_dict': optimizer.state_dict(),'epoch':epoch},os.path.join(output, f'checkpoint_{model_name}-{data_type}_{data_name}_{loss_name}.pth'))
    logger.info(f'best mAP:{best_mAP}')
    logger.info(f'best OF1:{best_of1}')

def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
        for i, (image_features, target,weight_00,weight_01,org_idx) in pbar:
            image_features = image_features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            weight_01 = weight_01.cuda(non_blocking=True)
            weight_00 = weight_00.cuda(non_blocking=True)
            org_idx = org_idx.cuda(non_blocking=True)
            optimizer.zero_grad()
            output = model(image_features)
            output = output.squeeze(1)
            loss = criterion(output,target,weight_01,weight_00,org_idx)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
            if (i+1) % print_freq == 0:
                logger.info(f'{epoch}/{epochs} {i}/{len(train_loader)} loss:{loss}')


def validate(val_loader, model, logger):
    global best_mAP
    model.eval()
    saved_data = []
    with torch.no_grad():
        for i, (image_features, target,weight_00,weight_01,org_idx) in tqdm(enumerate(val_loader),total=len(val_loader)):
            image_features = image_features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(image_features)
            output = output.squeeze(1)
            output_sm = nn.functional.sigmoid(output)

            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            saved_data.append(_item)

        saved_data = torch.cat(saved_data, 0).numpy()
        # saved_name = 'saved_data_tmp.txt'
        # output=f'/root/autodl-tmp/muti_label_classify/output/{data_name}_{data_type}_change'
        # np.savetxt(os.path.join(output, saved_name), saved_data)
        print("Calculating mAP:")
        metric_func = voc_mAP
        mAP, aps = metric_func(saved_data, num_classes,
                                   return_each=True)

        logger.info("  mAP: {}".format(mAP))
        metric_func = get_PR
        CF1, CP, CR, OF1, OP, OR = metric_func(saved_data, num_classes, output)
        logger.info("  OF1: {}".format(OF1 * 100))
        logger.info("  OP: {}".format(OP * 100))
        logger.info("  OR: {}".format(OR * 100))
        # logger.info("  aps: {}".format(np.array2string(aps, precision=5)))
        return mAP,OF1,OP,OR

if __name__ == '__main__':
    main()


