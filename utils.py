import os
import json

root_dir = os.path.dirname(os.path.abspath(__file__))
def json_write(js_name,img,ai_ans,pred_id,gt_id):
    if not os.path.exists(os.path.join(root_dir,f"{js_name}.json")):
        with open(f'{js_name}.json', 'w') as json_file:
            json.dump({}, json_file, indent=4)
    json_str = {f"{img}_ai": ai_ans,
                f"{img}_pred": pred_id,
                f"{img}_gt": gt_id}
    with open(f'{js_name}.json', 'r') as json_file:
        data = json.load(json_file)
    data.update(json_str)
    with open(f'{js_name}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def json_write_des(js_dir,des):
    if not os.path.exists(js_dir):
        with open(js_dir, 'w') as json_file:
            json.dump({}, json_file, indent=4)
    with open(js_dir, 'r') as json_file:
        data = json.load(json_file)
    data.update({'des':des})
    with open(js_dir, 'w') as json_file:
        json.dump(data, json_file, indent=4)