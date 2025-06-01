import json

data_name='' #coco ade20K food
json_dir=f'/root/{data_name}_train.json'
with open(json_dir, 'r') as f:
    js = json.load(f)
img_names = [img[:-3] for img in list(js.keys())[::3]]
modify_data = []
gt_data=[]
ai_data=[]
ai_num=0
gt_num=0
md_num=0
for img in img_names:
    pred_label = set(js[f'{img}_pred'])
    ai_num+=len(pred_label)
    gt_label = set(js[f'{img}_gt'])
    gt_num+=len(gt_label)
    u_label = list(gt_label&pred_label)
    if u_label:
        md_num+=len(list(gt_label&pred_label))
        modify_data.append({img:list(gt_label&pred_label)})
        gt_data.append({img:list(gt_label)})
        ai_data.append({img:list(pred_label)})
        
with open(f"/root/{data_name}_json/{data_name}_ai.json", 'w') as f:
    json.dump(ai_data, f)
with open(f"/root/{data_name}_json/{data_name}_modify.json", 'w') as f:
    json.dump(modify_data, f)
with open(f"/root/{data_name}_json/{data_name}_gt.json", 'w') as f:
    json.dump(gt_data, f)