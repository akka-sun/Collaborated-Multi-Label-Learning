import torchvision.transforms as transforms
from dataset.new_cocodataset import CoCoDataset
from dataset.new_ade20kdataset import Ade20kDataset
from dataset.new_fooddataset import FoodDataset
from randaugment import RandAugment
import os.path as osp


def get_datasets( data_name=None, train_img_dir=None, train_json_dir=None,train_ai_json_dir=None, val_img_dir=None,
                 val_json_dir=None,val_ai_json_dir=None, val=False):

    if 'coco' in data_name:
        # ! config your data path here.
        if val:
            val_dataset = CoCoDataset(
                image_dir=val_img_dir,
                modify_json_dir=val_json_dir,
                ai_json_dir = val_ai_json_dir,
                type='val'
            )
            return val_dataset
        train_dataset = CoCoDataset(
            image_dir=train_img_dir,
            modify_json_dir=train_json_dir,
            ai_json_dir = train_ai_json_dir,
        )
        val_dataset = CoCoDataset(
            image_dir=val_img_dir,
            modify_json_dir=val_json_dir,
            ai_json_dir = val_ai_json_dir,
            type='val'
        )

    elif 'ade20k' in data_name:
        if val:
            val_dataset = Ade20kDataset(
                image_dir=val_img_dir,
                modify_json_dir=val_json_dir,
                ai_json_dir = val_ai_json_dir,
                type='val'
            )
            return val_dataset
        train_dataset = Ade20kDataset(
            image_dir=train_img_dir,
            modify_json_dir=train_json_dir,
            ai_json_dir=train_ai_json_dir,
        )
        val_dataset = Ade20kDataset(
            image_dir=val_img_dir,
            modify_json_dir=val_json_dir,
            ai_json_dir=val_ai_json_dir,
            type='val'
        )
    elif 'food' in data_name:
        if val:
            val_dataset = FoodDataset(
                image_dir=val_img_dir,
                modify_json_dir=val_json_dir,
                ai_json_dir = val_ai_json_dir,
                type='val'
            )
            return val_dataset
        train_dataset = FoodDataset(
            image_dir=train_img_dir,
            modify_json_dir=train_json_dir,
            ai_json_dir=train_ai_json_dir
        )
        val_dataset = FoodDataset(
            image_dir=val_img_dir,
            modify_json_dir=val_json_dir,
            ai_json_dir=val_ai_json_dir,
            type='val'
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % data_name)

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
