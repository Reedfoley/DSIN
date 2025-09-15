import torch
from torch import nn
import math

class BertConfig(object):

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=512,
                 add_abs_pos_emb = False,
                 add_pos_enc = False):


        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.add_abs_pos_emb = add_abs_pos_emb
        self.add_pos_enc = add_pos_enc

BertLayerNorm = torch.nn.LayerNorm

def gelu(x):

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertAttention(nn.Module):
    def __init__(self, config):

        super().__init__()
        # 检查隐藏层大小是否是注意力头数量的整数倍
        if config.hidden_size % config.num_attention_heads != 0:
            # 如果不是整数倍，抛出 ValueError 异常
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        # 存储注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询矩阵，将输入的隐藏状态转换为查询向量
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # 初始化键矩阵，将输入的隐藏状态转换为键向量
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # 初始化值矩阵，将输入的隐藏状态转换为值向量
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 存储是否使用绝对位置嵌入的标志
        self.add_abs_pos_emb = config.add_abs_pos_emb
        # 如果使用绝对位置嵌入
        if self.add_abs_pos_emb:
            # 初始化绝对位置嵌入矩阵，形状为 (512, 每个注意力头的大小)
            self.abs_pos_emb = nn.Parameter(torch.randn(512, self.attention_head_size))
        # 初始化 Dropout 层，用于在注意力概率上应用 Dropout 操作
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)#(b, s,h, d) -> (b, h, s, d)

    def forward(self, hidden_states, context, attention_mask=None):
        """
        前向传播方法，计算注意力机制的输出。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状通常为 (b, s_q, d)，其中 b 是批次大小，s_q 是查询序列的长度，d 是隐藏层大小。
            context (torch.Tensor): 上下文信息，形状通常为 (b, s_k, d)，其中 s_k 是键值序列的长度。
            attention_mask (torch.Tensor, optional): 注意力掩码，用于屏蔽不需要关注的位置，形状通常为 (b, s_q, s_k)。默认为 None。

        Returns:
            torch.Tensor: 注意力机制的输出，形状为 (b, s_q, d)。
        """
        # 打印上下文和注意力掩码的形状，用于调试
        #print(context.size(),attention_mask.size())
        # 通过查询线性层将隐藏状态转换为查询向量
        mixed_query_layer = self.query(hidden_states) # (b, s_q, d)
        # 通过键线性层将上下文转换为键向量
        mixed_key_layer = self.key(context)
        # 通过值线性层将上下文转换为值向量
        mixed_value_layer = self.value(context)

        # 对查询向量进行形状变换，以适应多头注意力机制的计算需求
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # 对键向量进行形状变换，以适应多头注意力机制的计算需求
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # 对值向量进行形状变换，以适应多头注意力机制的计算需求
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 如果使用绝对位置嵌入
        if self.add_abs_pos_emb:
            # 截取上下文长度对应的位置嵌入
            pos_emb = self.abs_pos_emb[0:context.size(1),:]
            # 截取隐藏状态长度对应的位置嵌入
            pos_emb_q = self.abs_pos_emb[0:hidden_states.size(1),:]
            # 扩展位置嵌入以匹配查询向量的形状
            pos_emb_q = pos_emb_q.expand(query_layer.size(0), query_layer.size(1), -1, -1)
        
        # 计算查询向量和键向量的点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape is (b, h, s_q, s_k)
        # 如果使用绝对位置嵌入
        if self.add_abs_pos_emb:
            # 计算查询向量加上位置嵌入后与位置嵌入的点积，得到位置注意力分数
            attention_pos_scores = torch.matmul(query_layer+pos_emb_q, pos_emb.transpose(-1, -2))
            # 将原始注意力分数和位置注意力分数相加，并除以注意力头大小的平方根进行缩放
            attention_scores = (attention_scores+attention_pos_scores) / math.sqrt(self.attention_head_size)
        else:
            # 若不使用绝对位置嵌入，直接将原始注意力分数除以注意力头大小的平方根进行缩放
            attention_scores = attention_scores/ math.sqrt(self.attention_head_size)
            
        # 如果提供了注意力掩码
        if attention_mask is not None:
            # 在维度 1 和 2 上增加维度，以匹配注意力分数的形状
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 扩展注意力掩码以匹配注意力分数的形状
            attention_mask = attention_mask.expand((-1,attention_scores.size(1),attention_scores.size(2),-1))
            # 将注意力掩码中值为 0 的位置替换为负无穷大
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            # 将注意力掩码中值为 1 的位置替换为 0
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
            # 打印注意力掩码和注意力分数的形状，用于调试
            #print(attention_mask.size())
            #print(attention_scores.size())
            # 将注意力分数和注意力掩码相加，屏蔽不需要关注的位置
            attention_scores = attention_scores + attention_mask
            
        # 对注意力分数进行 Softmax 操作，将其转换为注意力概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 对注意力概率应用 Dropout 操作，防止过拟合
        attention_probs = self.dropout(attention_probs)

        # 计算注意力概率和值向量的矩阵乘法，得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer) # shape is (b, h, s_q, d)
        # 交换维度，将形状从 (b, h, s_q, d) 转换为 (b, s_q, h, d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # shape is (b, s_q, h, d)
        # 计算新的形状，将注意力头的维度合并
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 调整上下文层的形状，将注意力头的维度合并
        context_layer = context_layer.view(*new_context_layer_shape)
        # 返回注意力机制的输出
        return context_layer


class BertAttOutput(nn.Module):
    """
    该类用于处理注意力机制的输出，对其进行线性变换、Dropout 操作和 Layer Normalization。
    """
    def __init__(self, config):
        """
        初始化 BertAttOutput 类的实例。

        Args:
            config (BertConfig): 包含模型配置的对象，包含如隐藏层大小、Dropout 概率等参数。
        """
        # 调用父类 nn.Module 的构造函数
        super(BertAttOutput, self).__init__()
        # 初始化线性层，将输入的隐藏状态进行线性变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 Layer Normalization 层，用于对输入的隐藏状态进行归一化处理
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 初始化 Dropout 层，用于防止模型过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        前向传播方法，对注意力机制的输出进行处理。

        Args:
            hidden_states (torch.Tensor): 注意力机制的输出，形状通常为 (b, s, d)，其中 b 是批次大小，s 是序列长度，d 是隐藏层大小。
            input_tensor (torch.Tensor): 输入的张量，形状通常为 (b, s, d)。

        Returns:
            torch.Tensor: 处理后的隐藏状态，形状为 (b, s, d)。
        """
        # 通过线性层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态应用 Dropout 操作，防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的隐藏状态与输入张量相加，并进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
    
class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor) # attention_output = self.output(output, input_tensor)

        return attention_output

class AttLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Self-attention Layers
        self.text_self_att = BertSelfattLayer(config)
        self.audio_self_att = BertSelfattLayer(config)
        self.vision_self_att = BertSelfattLayer(config)
        
        # Unimodal output
        self.text_unimodal_inter = BertIntermediate(config)
        self.text_unimodal_output = BertOutput(config)
        self.audio_unimodal_inter = BertIntermediate(config)
        self.audio_unimodal_output = BertOutput(config)
        self.vision_unimodal_inter = BertIntermediate(config)
        self.vision_unimodal_output = BertOutput(config)
        
        # Cross-attention Layer
        self.text_cross_att = BertCrossattLayer(config)
        self.audio_cross_att = BertCrossattLayer(config)
        self.vision_cross_att = BertCrossattLayer(config)
        
        # Output
        self.text_inter = BertIntermediate(config)
        self.text_output = BertOutput(config)
        self.audio_inter = BertIntermediate(config)
        self.audio_output = BertOutput(config)
        self.vision_inter = BertIntermediate(config)
        self.vision_output = BertOutput(config)
       
    def forward(self, text_inputs, text_unimodal_inputs, text_mask, audio_inputs, audio_unimodal_inputs, audio_mask, vision_inputs, vision_unimodal_inputs, vision_mask):
        
        text = text_inputs
        audio = audio_inputs
        vision = vision_inputs
        
        text_unimodal = text_unimodal_inputs
        audio_unimodal = audio_unimodal_inputs
        vision_unimodal = vision_unimodal_inputs
        
        text_self_att = self.text_self_att(text, text_mask)
        audio_self_att = self.audio_self_att(audio, audio_mask)
        vision_self_att = self.vision_self_att(vision, vision_mask)
        
        # Unimodal output
        text_unimodal_self_att = self.text_self_att(text_unimodal, text_mask)
        audio_unimodal_self_att = self.audio_self_att(audio_unimodal, audio_mask)
        vision_unimodal_self_att = self.vision_self_att(vision_unimodal, vision_mask)
        
        text_unimodal_inter = self.text_unimodal_inter(text_unimodal_self_att)
        audio_unimodal_inter = self.audio_unimodal_inter(audio_unimodal_self_att)
        vision_unimodal_inter = self.vision_unimodal_inter(vision_unimodal_self_att)
        
        text_unimodal_output = self.text_unimodal_output(text_unimodal_inter, text_unimodal_self_att)
        audio_unimodal_output = self.audio_unimodal_output(audio_unimodal_inter, audio_unimodal_self_att) 
        vision_unimodal_output = self.vision_unimodal_output(vision_unimodal_inter, vision_unimodal_self_att)
        # end unimodal output
        
        text_cross_att = self.text_cross_att(text_self_att, torch.cat((audio_self_att, vision_self_att), dim=1),ctx_att_mask=torch.cat((audio_mask, vision_mask), dim=1))
        audio_cross_att = self.audio_cross_att(audio_self_att, text_self_att, ctx_att_mask=text_mask)
        vision_cross_att = self.vision_cross_att(vision_self_att, text_self_att, ctx_att_mask=text_mask)
        
        # Output
        text_inter = self.text_inter(text_cross_att)
        audio_inter = self.audio_inter(audio_cross_att)
        vision_inter = self.vision_inter(vision_cross_att)
        
        text_output = self.text_output(text_inter, text_cross_att)
        audio_output = self.audio_output(audio_inter, audio_cross_att)
        vision_output = self.vision_output(vision_inter, vision_cross_att)
        
        return text_output, text_unimodal_output, audio_output, audio_unimodal_output, vision_output, vision_unimodal_output

        
        
        
        
        
        