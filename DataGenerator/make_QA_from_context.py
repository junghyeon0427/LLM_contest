# -*- coding: utf-8 -*-
import json
from glob import glob
import os.path as osp
from utils import Dataset, gpt_call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-num', type=int, required=True)
args = parser.parse_args()
total_data = list()
error_data = list()


def save_json(save_path, not_error:bool):
    # save total_data in json format
    with open(save_path, 'w') as f:
        if not_error:
            json.dump(total_data, f, ensure_ascii=False)
        else:
            json.dump(error_data, f, ensure_ascii=False)


if __name__ == '__main__':

    num = args.num

    total_dataset = list()
    for json_file in glob('total_data/*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list)
            data = list(set(data))  # 중복 제거
            total_dataset.extend(data)

    dataset = Dataset(list(set(total_dataset)), num=num)  

    flag = True
    idx = 0

    while True:
        context = dataset.next()
        if context == None:
            print(f'Complte')
            exit()
        answer, success = gpt_call(context, flag)
        
        if answer == None and success == False:
            continue
        
        idx += 1
        if idx % 3 == 0:
            flag = False
        else: 
            flag = True
            
        if success:
            total_data.append(answer)
            save_json(save_path=osp.join('GPT_data', f'{num}_total_data.json'), not_error=True)
        else:
            error_data.append(answer)
            save_json(save_path=osp.join('GPT_data', f'{num}_error_data.json'), not_error=False)
