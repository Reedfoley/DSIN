import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel
import torch.nn.functional as F
from utils.atten_encoder import MMELayer, BertConfig
# from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 自定义注意力扩展层
class AttnExpandLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # 分步执行以支持Sequential
        inputs = F.relu(self.linear1(inputs)) # 扩展
        attn = self.attention(inputs) # 生成注意力权重
        inputs = inputs * attn # 特征增强
        inputs = self.linear2(inputs) # 扩展
        inputs = self.layer_norm(inputs) # 归一化
        return self.dropout(inputs) # 正则化
  
class mme(nn.Module):            
    def __init__(self, config):        
        super().__init__()

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained('premodel\chinese-roberta-wwm-ext')
        
        self.expand_audio= nn.Sequential(AttnExpandLayer(33, 256, 768, config.dropout))
        self.expand_vision= nn.Sequential(AttnExpandLayer(709, 768, 768, config.dropout))
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.vision_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)       


        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers)
        self.MME_layers = nn.ModuleList(
            [MMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

        # fused method V2
        self.text_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU()
        )
        self.audio_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU()
        )
        
        # output layers for each single modality
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
          )
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
          )
        

        self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768*3, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        
    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'vision':
            embedding_layer = self.vision_cls_emb
        
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask, vision_inputs, vision_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        text_features = raw_output.last_hidden_state
        
        # Attention-Based Modal Enhancement Module 
        audio_features = self.expand_audio(audio_inputs) # [batch_size, seq_len, 1024]
        vision_features = self.expand_vision(vision_inputs) # [batch_size, seq_len, 1024]
        
        # 添加 CLS 标记
        text_inputs, text_attn_mask = self.prepend_cls(text_features, text_mask, 'text') # [batch_size, text_seq_len+1, hidden_size], [batch_size, text_seq_len+1]
        audio_inputs, audio_attn_mask = self.prepend_cls(audio_features, audio_mask, 'audio') # [batch_size, audio_seq_len+1, hidden_size], [batch_size, audio_seq_len+1]
        vision_inputs, vision_attn_mask = self.prepend_cls(vision_features, vision_mask, 'vision') # [batch_size, vision_seq_len+1, hidden_size], [batch_size, vision_seq_len+1]  
        
        text_unimodal = text_inputs
        audio_unimodal = audio_inputs
        vision_unimodal = vision_inputs       

        # MME
        for layer_module in self.MME_layers:
            text_inputs, text_unimodal, audio_inputs, audio_unimodal, vision_inputs, vision_unimodal = layer_module(text_inputs, text_unimodal, text_attn_mask, audio_inputs, audio_unimodal, audio_attn_mask, vision_inputs, vision_unimodal, vision_attn_mask)
        
        text_CLS = text_unimodal[:,0,:]
        audio_CLS = audio_unimodal[:,0,:]
        vision_CLS = vision_unimodal[:,0,:]
        
        fused_hidden_states = torch.cat((text_inputs[:,0,:], audio_inputs[:,0,:], vision_inputs[:,0,:]), dim=1) 
        
        T_output = self.T_output_layers(text_CLS)
        A_output = self.A_output_layers(audio_CLS)
        V_output = self.V_output_layers(vision_CLS)
        
        fused_output = self.fused_output_layers(fused_hidden_states) 
        
        return {
                'T': T_output, 
                'A': A_output, 
                'V': V_output,
                'M': fused_output
        }
        
    


