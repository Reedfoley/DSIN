import torch
from torch import nn
import transformers
import torchaudio
import pickle
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import string

class Dataset_mosi(torch.utils.data.Dataset):
    
    def __init__(self, pkl_path, mode):

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)    
            
        raw_texts = data[mode]['raw_text']
        processed_texts = [text[0] + text[1:].lower() for text in raw_texts]
        self.texts = processed_texts
    
        # 初始化分词器，使用预训练的roberta-large模型
        self.tokenizer = AutoTokenizer.from_pretrained("premodel/roberta-large")

        self.visions = data[mode]['vision'].astype(np.float32)
        self.audios = data[mode]['audio'].astype(np.float32)
        
        self.targets_M = data[mode]['regression_labels'].astype(np.float32)

    def __getitem__(self, index):

        # 加载当前索引对应的文本
        text = str(self.texts[index])             
        
        # 对当前文本进行分词处理
        tokenized_text = self.tokenizer(
                text,
                # 最大长度设置为96
                max_length = 96,
                # 填充到指定的最大长度
                padding = "max_length",
                # 截断到指定的最大长度
                truncation = True,
                # 是否插入特殊标记，如 [CLS], [SEP], <s> 等
                add_special_tokens = True,
                # 返回注意力掩码
                return_attention_mask = True
            )
        
        audio_masks = torch.ones(50, dtype=torch.float32)
        vision_masks = torch.ones(50, dtype=torch.float32)
        
        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
                # audio
                "audio_inputs": torch.Tensor(self.audios[index]),
                "audio_masks": audio_masks,
                # vision
                "vision_inputs": torch.Tensor(self.visions[index]),    
                "vision_masks": vision_masks,          
                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                }
    
    def __len__(self):
        return len(self.targets_M)
    
 

def data_loader(batch_size, dataset):
    if dataset == 'mosi':
        pkl_path = 'data/MOSI/feature.pkl'
        train_data = Dataset_mosi(pkl_path, 'train')
        test_data = Dataset_mosi(pkl_path, 'test')
        val_data = Dataset_mosi(pkl_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader  
    elif dataset == 'mosei':
        pkl_path = 'data/MOSEI/feature.pkl'
        train_data = Dataset_mosi(pkl_path, 'train')
        test_data = Dataset_mosi(pkl_path, 'test')
        val_data = Dataset_mosi(pkl_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader 
    
