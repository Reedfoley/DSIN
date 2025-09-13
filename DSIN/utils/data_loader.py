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

class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, pkl_path, mode):       
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # store labels
        self.targets_M = data[mode]['regression_labels'].astype(np.float32)
        self.targets_T = data[mode]['regression_labels_T'].astype(np.float32)
        self.targets_A = data[mode]['regression_labels_A'].astype(np.float32)
        self.targets_V = data[mode]['regression_labels_V'].astype(np.float32)
        
        raw_texts = data[mode]['raw_text']
        processed_texts = [text[0] + text[1:].lower() for text in raw_texts]
        self.texts = processed_texts
        
        self.tokenizer = AutoTokenizer.from_pretrained("premodel/chinese-roberta-wwm-ext")
        
        self.visions = data[mode]['vision'].astype(np.float32)
        self.audios = data[mode]['audio'].astype(np.float32)
        
        
    def __getitem__(self, index):
       # extract text features
        text = str(self.texts[index])         
        tokenized_text = self.tokenizer(
            text,            
            max_length = 64,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )    
        
        audio_masks = torch.ones(400, dtype=torch.float32)
        vision_masks = torch.ones(55, dtype=torch.float32)                   
                
            
        return { # text
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                # audio
                "audio_inputs": torch.Tensor(self.audios[index]),
                "audio_masks": audio_masks,
                # vision
                "vision_inputs": torch.Tensor(self.visions[index]),    
                "vision_masks": vision_masks,   
                 # labels
                "target": {
                    "M": self.targets_M[index],
                    "T": self.targets_T[index],
                    "A": self.targets_A[index],
                    "V": self.targets_V[index],
                }
                }
    
    def __len__(self):
        return len(self.targets_M)

class Dataset_mosi(torch.utils.data.Dataset):
    
    def __init__(self, pkl_path, mode, text_context_length=2, audio_context_length=1):

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
    

def collate_fn_sims(batch):   
    text_tokens = []  
    text_masks = []
    audio_inputs = []  
    audio_masks = []
    vision_inputs = []  
    vision_masks = []
    
    targets_M = []
    targets_T = []
    targets_A = []
    targets_V = []
   
    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        #audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])
        #vision
        vision_inputs.append(batch[i]['vision_inputs'])
        vision_masks.append(batch[i]['vision_masks'])        

       # labels
        targets_M.append(batch[i]['target']['M'])
        targets_T.append(batch[i]['target']['T'])
        targets_A.append(batch[i]['target']['A'])
        targets_V.append(batch[i]['target']['V'])
        
       
    return {
            # text
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "text_masks": torch.tensor(text_masks, dtype=torch.long),           
            # audio
            "audio_inputs": torch.stack(audio_inputs),
            "audio_masks": torch.stack(audio_masks),
            # vision
            "vision_inputs": torch.stack(vision_inputs),
            "vision_masks": torch.stack(vision_masks),
            # labels
            "targets": {
                    "M": torch.tensor(targets_M, dtype=torch.float32),
                    "T": torch.tensor(targets_T, dtype=torch.float32),
                    "A": torch.tensor(targets_A, dtype=torch.float32),
                    "V": torch.tensor(targets_V, dtype=torch.float32)
                }
            }  

def data_loader(batch_size, dataset, text_context_length=2, audio_context_length=1):
    if dataset == 'mosi':
        pkl_path = 'data/MOSI/feature.pkl'
        train_data = Dataset_mosi(pkl_path, 'train', text_context_length=text_context_length, audio_context_length=audio_context_length)
        test_data = Dataset_mosi(pkl_path, 'test', text_context_length=text_context_length, audio_context_length=audio_context_length)
        val_data = Dataset_mosi(pkl_path, 'valid', text_context_length=text_context_length, audio_context_length=audio_context_length)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader  
    elif dataset == 'mosei':
        pkl_path = 'data/MOSEI/feature.pkl'
        train_data = Dataset_mosi(pkl_path, 'train', text_context_length=text_context_length, audio_context_length=audio_context_length)
        test_data = Dataset_mosi(pkl_path, 'test', text_context_length=text_context_length, audio_context_length=audio_context_length)
        val_data = Dataset_mosi(pkl_path, 'valid', text_context_length=text_context_length, audio_context_length=audio_context_length)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader 
    
    else:
        pkl_path = 'data/SIMS/feature.pkl'
        train_data = Dataset_sims(pkl_path, 'train')
        test_data = Dataset_sims(pkl_path, 'test')
        val_data = Dataset_sims(pkl_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        return train_loader, test_loader, val_loader