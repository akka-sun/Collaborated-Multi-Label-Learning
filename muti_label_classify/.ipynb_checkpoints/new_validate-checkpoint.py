import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

from tqdm import tqdm

from models.clip_muti import CLIP
from dataset.get_new_dataset import get_datasets
from utils.logger import Logger
from utils.metrics import voc_mAP,get_PR

batch_size = 2048
img_size=512
epochs=80
print_freq=100

data_name='coco' #coco coco_ai ade20k
if data_name == 'coco':
    num_classes = 80
    val_img_dir = '/root/autodl-tmp/val2017'
elif data_name == 'ade20k':
    num_classes = 150
    val_img_dir = '/root/autodl-tmp/ADEChallengeData2016/images/validation'
data_type='modify'
model_name='clip'
val_json_dir=f'/root/autodl-tmp/{data_name}_json/{data_name}_val.json'
val_ai_json_dir = f'/root/autodl-tmp/{data_name}_json/{data_name}_val.json'

lr=1e-3
weight_decay=1e-4

output='/root/autodl-tmp/muti_label_classify/output'
checkpoint_path=f'/root/autodl-tmp/muti_label_classify/output/{data_name}_modify_modify_change/checkpoint_clip-modify_{data_name}_new.pth'

def main():
    model = CLIP(num_classes)
    model = model.cuda()
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict['state_dict'])
    val_dataset = get_datasets(data_name,val_img_dir=val_img_dir,val_json_dir=val_json_dir,val_ai_json_dir=val_json_dir,val=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger=Logger(f'{data_name}-{data_type}_val_new').logger
    validate(val_loader,model,logger)


def validate(val_loader, model, logger):
    model.eval()
    saved_data = []
    with torch.no_grad():
        for i, (image_features, target,weight_00,weight_01,org_idx) in tqdm(enumerate(val_loader),total=len(val_loader)):
            image_features = image_features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(image_features)
            output = output.squeeze()
            output_sm = nn.functional.sigmoid(output)

            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            saved_data.append(_item)

        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = f'saved_{data_name}_val.txt'
        output=f'/root/autodl-tmp/muti_label_classify/output/{data_name}_{data_type}'
        np.savetxt(os.path.join(output, saved_name), saved_data)
        print("Calculating:")
        metric_func = voc_mAP
        mAP, aps = metric_func([os.path.join(output, saved_name)], num_classes,
                                   return_each=True)

        logger.info("  mAP: {}".format(mAP))
        logger.info("  aps: {}".format(np.array2string(aps, precision=5)))

        metric_func = get_PR
        CF1,CP,CR,OF1,OP,OR=metric_func([os.path.join(output, saved_name)], num_classes,output)
        logger.info("  CF1: {}".format(CF1 * 100))
        logger.info("  CP: {}".format(CP * 100))
        logger.info("  CR: {}".format(CR * 100))
        logger.info("  OF1: {}".format(OF1 * 100))
        logger.info("  OP: {}".format(OP * 100))
        logger.info("  OR: {}".format(OR * 100))

if __name__ == '__main__':
    main()


