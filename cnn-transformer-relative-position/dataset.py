
from utils import ROOT_DIR, IMG_DIR, IMG_TRANSFORM
from copy import deepcopy
import random
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

DIGITS= [str(i) for i in range(0, 10)]
NULL = '<NULL>'
SYMBOLS = DIGITS + ['s'] + ['e'] + [NULL]
SYMBOLS_AND_OPS = SYMBOLS + ['+'] + ['-'] + ['*'] + ['/'] + ['!'] + ['('] + [')']
SYM2ID = lambda x: SYMBOLS.index(x)
SYMOP2ID = lambda x: SYMBOLS_AND_OPS.index(x)


class HINT(Dataset):
    def __init__(self, split='train', n_sample_zero_res=None):
        super(HINT, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset = json.load(open(ROOT_DIR + 'expr_%s.json'%split))        
        
        if n_sample_zero_res is not None:
            samples_non_zero = [x for x in self.dataset if len(x['expr']) == 1 or (x['res'] != 0)]
            samples_zero = [x for x in self.dataset if len(x['expr']) > 1 and (x['res'] == 0)]
            samples_zero = random.sample(samples_zero, int(len(samples_zero) * n_sample_zero_res))
            self.dataset = samples_non_zero + samples_zero

        for x in self.dataset:
            x['len'] = len(x['expr'])
        
        self.img_transform = IMG_TRANSFORM
        self.valid_ids = list(range(len(self.dataset)))

        # dataset statistics, used to filter samples
        len2ids = {}
        for i, x in enumerate(self.dataset):
            l = len(x['img_paths'])
            if l not in len2ids:
                len2ids[l] = []
            len2ids[l].append(i)
        self.len2ids = len2ids

        res2ids = {}
        for i, x in enumerate(self.dataset):
            l = x['res']
            if l not in res2ids:
                res2ids[l] = []
            res2ids[l].append(i)
        self.res2ids = res2ids

    def __getitem__(self, index):
        index = self.valid_ids[index]
        sample = deepcopy(self.dataset[index])
        img_seq = []
        item = dict()
        for img_path in sample['img_paths']:
            img = Image.open(IMG_DIR+img_path).convert('L')
            img = self.img_transform(img)
            img_seq.append(img)
        label_seq = str(sample['res'])
        label_list = [SYM2ID(x) for x in label_seq]
        label_list = [SYM2ID('s')] + label_list + [SYM2ID('e')]
        expr_seq = str(sample['expr'])
        expr_list = [SYMOP2ID(x) for x in expr_seq]
        expr_list = [SYMOP2ID('s')] + expr_list + [SYMOP2ID('e')]
        item['img_seq'] = img_seq
        item['label_seq'] = label_list
        item['expr_seq'] = expr_list
        item['res'] = sample['res']
        item['len'] = sample['len']
        return item
            
    def __len__(self):
        return len(self.valid_ids)

def HINT_collate(batch):
    max_len_i = np.max([x['len'] for x in batch])
    max_len_o = np.max([len(x['label_seq']) for x in batch])
    max_len_e = np.max([len(x['expr_seq']) for x in batch])
    zero_img = torch.zeros_like(batch[0]['img_seq'][0]) 

    for sample in batch:
        sample['img_seq'] += [zero_img] * (max_len_i - sample['len'])
        sample['img_seq'] = torch.stack(sample['img_seq'])

        sample['mask'] = [1] * sample['len'] + [0] * (max_len_i - sample['len'])
        sample['mask'] = torch.tensor(sample['mask'])

        sample['expr_seq'] += [SYMOP2ID(NULL)] * (max_len_e - (sample['len'] + 2))
        sample['expr_seq'] = torch.tensor(sample['expr_seq'])    

        sample['expr_mask'] = [1] * (sample['len'] + 2) + [0] * (max_len_e - (sample['len'] + 2))
        sample['expr_mask'] = torch.tensor(sample['expr_mask'])

        sample['label_seq'] += [SYM2ID(NULL)] * (max_len_o - len(str(sample['res'])))
        sample['label_seq'] = torch.tensor(sample['label_seq'])
        
    batch = default_collate(batch)

    return batch

if __name__ == '__main__':
    val_set = HINT('val')
    val_loader = DataLoader(val_set, batch_size=32,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)

    print(next(iter(val_loader)))
