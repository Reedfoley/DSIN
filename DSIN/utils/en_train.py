import torch
from torch import nn
from tqdm import tqdm
import random
import numpy as np

from utils.data_loader import data_loader
from utils.en_model import DSIN
from utils.metricsTop import MetricsTop

# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

class EnConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'regression',
                loss_weights = {
                    'M':1,
                    'T':1,
                    'A':1,
                    'V':1
                },
                 model_save_path = 'checkpoint/',
                 learning_rate = 1e-5,
                 epochs = 20,
                 dataset_name = 'mosei',
                 early_stop = 8,
                 seed = 0,
                 dropout=0.3,
                 batch_size = 16,
                 multi_task = True,
                 num_hidden_layers = 1,
                 tasks = 'M',   # 'M' or 'MTA',
                ):

        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks

class EnTrainer():
    def __init__(self, config):
        """
        初始化 EnTrainer 类的实例。

        Args:
            config (EnConfig): 包含训练配置的对象，包含如训练模式、损失权重、学习率等参数。
        """
        # 存储传入的配置对象，方便后续方法使用
        self.config = config
        # 根据配置的训练模式选择合适的损失函数
        # 如果训练模式为回归（'regression'），则使用 L1 损失函数
        # 否则，使用交叉熵损失函数
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        # 根据训练模式和数据集名称获取相应的评估指标
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        # 存储配置中指定的训练任务
        self.tasks = config.tasks
        
    def do_train(self, model, data_loader):
        """
        执行模型的训练过程，遍历数据加载器中的所有批次数据，进行前向传播、损失计算、反向传播和参数更新。

        Args:
            model (torch.nn.Module): 要训练的模型。
            data_loader (torch.utils.data.DataLoader): 提供训练数据的迭代器。

        Returns:
            float: 该训练周期的平均训练损失。
        """
        # 将模型设置为训练模式，启用如 Dropout、BatchNorm 等训练时使用的特殊层
        model.train()
        # 初始化 AdamW 优化器，用于更新模型的参数，学习率从配置中获取
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        # 初始化总损失为 0，用于累加每个批次的损失
        total_loss = 0
        # 遍历数据加载器中的每个批次，使用 tqdm 显示进度条
        # Loop over all batches.
        for batch in tqdm(data_loader):
            # 从批次中提取文本输入和对应的掩码，并移动到指定设备（如 GPU）
            text_inputs = batch["text_tokens"].to(device) # [batch_size, max_len]
            text_mask = batch["text_masks"].to(device)

            # 从批次中提取音频输入和对应的掩码，并移动到指定设备
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            vision_inputs = batch["vision_inputs"].to(device)
            vision_mask = batch["vision_masks"].to(device)

            # 从批次中提取目标值，并调整形状为 [batch_size, 1]，然后移动到指定设备
            targets = batch["targets"].to(device).view(-1, 1)

            # 清零优化器中的梯度，避免梯度累积
            optimizer.zero_grad()

            outputs = model(text_inputs, text_mask, audio_inputs, audio_mask, vision_inputs, vision_mask)

            # 计算训练损失
            # Compute the training loss.
            if self.config.multi_task:
                # 初始化总损失为 0
                loss = 0.0
                # 遍历每个任务，计算并累加每个任务的加权损失
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
    #                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
                # 累加当前批次的总损失到总损失中
                total_loss += loss.item() * text_inputs.size(0)
            else:
                # 单任务训练时，只计算 'M' 任务的损失
                loss = self.criterion(outputs['M'], targets)
                # 累加当前批次的损失到总损失中
                total_loss += loss.item() * text_inputs.size(0)

            # 反向传播计算梯度
            loss.backward()
            # 根据计算得到的梯度更新模型的参数
            optimizer.step()

        # 计算整个训练集的平均损失，并保留四位小数
        total_loss = round(total_loss / len(data_loader.dataset), 4)
#         print('TRAIN'+" >> loss: ",total_loss)
        # 返回平均损失
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()   # Put the model in eval mode.
        if self.config.multi_task:
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            total_loss = 0
            val_loss = {
                'M':0,
                'T':0,
                'A':0,
                'V':0
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader):                    # Loop over all batches.
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                vision_inputs = batch["vision_inputs"].to(device)
                vision_mask = batch["vision_masks"].to(device)
                
                targets = batch["targets"].to(device).view(-1, 1)

                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask, vision_inputs, vision_mask)
                
                # Compute loss.
                if self.config.multi_task:
                    loss = 0.0         
                    for m in self.tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                    total_loss += loss.item()*text_inputs.size(0)
                    # add predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

        if self.config.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A'], "  V_loss: ", val_loss['V'])

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss 
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
        
        return eval_results

def EnRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,)
    
    model = DSIN(config).to(device) 
        
    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader,"VAL")

        if eval_results['Loss']<lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), config.model_save_path+f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth')
            best_epoch = epoch
        if eval_results['Has0_acc_2']>=highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            torch.save(model.state_dict(), config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth'))        
    test_results_loss = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))

    model.load_state_dict(torch.load(config.model_save_path+f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth'))
    test_results_acc = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))    

