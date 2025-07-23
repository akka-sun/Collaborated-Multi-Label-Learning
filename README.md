# Collaborated-Multi-Label-Learning
## Preparing Dataset
### COCO
```
  wget -c http://images.cocodataset.org/zips/train2017.zip
  wget -c http://images.cocodataset.org/zips/val2017.zip
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip train2017.zip
  unzip val2017.zip
  unzip annotations_trainval2017.zip
```
### ADE20K
[here](https://ai-studio-online.bj.bcebos.com/v1/fc797adb86ea40418938c95a9291a249ed160e826a5844c087b2b46a89e2e5a7?responseContentDisposition=attachment%3Bfilename%3DADEChallengeData2016.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2025-07-23T04%3A20%3A21Z%2F60%2F%2F582c8ca6fdd1af1cfb32dd31950c218f4a4456cc5304d064f90bc36863404044) for download
### FOODSEG
see <https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1>
## Get CML 
### Preparing VLM api
prepare VLM api and fill in the api in `vlm_api.py`
### Generate CML
run `coco_data.py` `ade20k_data.py` `food_data.py` to generate CML
## Train the Model
### Generate txt file
use [TagCLIP](https://github.com/linyq2117/TagCLIP) to generate txt file
