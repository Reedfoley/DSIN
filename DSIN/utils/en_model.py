import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel

from utils.atten_encoder import AttLayer, BertConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 自定义注意力扩展层
# 自定义LSTM扩展层
class LSTMExpandLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        # 使用双向LSTM捕获序列信息
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 线性层将LSTM输出映射到目标维度
        self.linear = nn.Linear(hidden_dim * 2, output_dim) 
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        # 注意力机制用于特征增强
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        # 输入形状: [batch_size, seq_len, input_dim]
        # LSTM处理
        lstm_output, _ = self.lstm(inputs)
        # 生成注意力权重
        attn_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        # 应用注意力权重
        attended_output = lstm_output * attn_weights
        # 全局平均池化以处理可变长度序列
        attended_output = torch.mean(attended_output, dim=1)  # [batch_size, hidden_dim*2]
        # 映射到目标维度
        outputs = self.linear(attended_output)  # [batch_size, output_dim]
        # 归一化和正则化
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        # 为了保持与原始层相同的输出形状，我们需要调整维度
        # 假设inputs的形状是 [batch_size, seq_len, input_dim]
        # 我们需要返回 [batch_size, seq_len, output_dim]
        # 因此我们将池化后的特征复制到每个时间步
        batch_size, seq_len, _ = inputs.shape
        outputs = outputs.unsqueeze(1).expand(batch_size, seq_len, output_dim)
        return outputs

class DSIN(nn.Module):
    def __init__(self, config):

        super().__init__()
        # 加载预训练的 RoBERTa 模型
        self.roberta_model = RobertaModel.from_pretrained('premodel/roberta-large')
        
        if config.dataset_name == 'mosi':
            self.expand_audio = nn.Sequential(LSTMExpandLayer(5, 256, 1024, config.dropout))
            self.expand_vision = nn.Sequential(LSTMExpandLayer(20, 256, 1024, config.dropout))
        else:
            self.expand_audio = nn.Sequential(LSTMExpandLayer(74, 256, 1024, config.dropout))
            self.expand_vision = nn.Sequential(LSTMExpandLayer(35, 256, 1024, config.dropout))
            
        # CLS嵌入层
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024) 
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        self.vision_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        
        # 配置 BERT 模型的参数
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=1024, intermediate_size=4096, num_attention_heads=16)
        # 创建 ATT 层列表
        self.att_layers = nn.ModuleList(
            [AttLayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )
        
        

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
        )

        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
        )

        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
        )
        
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*3, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )
        
    def prepend_cls(self, inputs, masks, layer_name):
        """
        为输入序列添加 CLS 标记。

        Args:
            inputs (torch.Tensor): 输入序列的张量。
            masks (torch.Tensor): 输入序列的掩码张量。
            layer_name (str): 层的名称，用于选择相应的嵌入层。

        Returns:
            tuple: 包含添加 CLS 标记后的输入序列和掩码的元组。
        """
        # 根据层名称选择相应的嵌入层
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'vision':
            embedding_layer = self.vision_cls_emb
        # 创建一个包含单个元素 0 的长整型张量，并将其移动到与输入相同的设备上
        index = torch.LongTensor([0]).to(device=inputs.device) # Shape is [1]
        # 通过嵌入层获取 CLS 标记的嵌入表示
        cls_emb = embedding_layer(index) # Shape is [1, 1, 1024]
        # 将 CLS 标记的嵌入表示扩展为与输入序列的批次大小相同
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2)) # Shape is [batch_size, 1, 1024]
        # 将 CLS 标记的嵌入表示添加到输入序列的开头
        outputs = torch.cat((cls_emb, inputs), dim=1) # Shape is [batch_size, seq_len+1, 1024]
        
        # 创建一个全为 1 的张量，用于表示 CLS 标记的掩码
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device) # Shape is [batch_size, 1]
        # 将 CLS 标记的掩码添加到输入序列的掩码开头
        masks = torch.cat((cls_mask, masks), dim=1) # Shape is [batch_size, seq_len+1]
        return outputs, masks
        
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask, vision_inputs, vision_mask):
        
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True) 
        text_features = raw_output.last_hidden_state # [batch_size, seq_len, 1024]
        
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
        for layer_module in self.att_layers:
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
        
        
        
        
        