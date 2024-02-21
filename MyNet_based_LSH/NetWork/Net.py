import numpy as np
import pandas as pd
import GenerateDataAndPlot as gp
import auto as at
from layer import Connected, Convolution, Pooling
import os
from random import randint
import json
from tqdm import tqdm
import time
from LogAndLoss import Loss_Saver, get_logger


class Net:
    def __init__(self) -> None:
        self.model_path = 'C:/Users/szfmsmdx/Desktop/毕业论文/MyNet_based_LSH/NetWork/Model/' + time.strftime("%Y-%m-%d_%H：%M：%S")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.log_path = 'C:/Users/szfmsmdx/Desktop/毕业论文/MyNet_based_LSH/NetWork/Log/' + time.strftime("%Y-%m-%d_%H：%M：%S") + '.txt'
        self.log = get_logger(self.log_path)

        self.layers = []
        self.learning_rate = 0.1
        self.loss_log = []
        self.loss_function = at.MSEOp()
        self.optimizer = None

        # train 完更新
        self.epoch = 1
        self.batch_size = 1           
        self.curloss = None         # 当前损失

    def set(self, learning_rate, loss_function, optimizer = None, epoch = None, batch_size = None):
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        if loss_function == 'MSE':
            self.loss_function = at.MSEOp()
        elif loss_function == 'CrossEntropy':
            self.loss_function = at.CrossEntropyOp()
        else:
            self.log.error('loss_function not found')
            raise ValueError('loss_function not found')
        if optimizer != None and optimizer not in ['SGD', 'LGD']:
            self.log.error('optimizer not found')
            raise ValueError('optimizer not found')
        else:
            self.optimizer = optimizer
        if epoch != None:
            self.epoch = epoch
        if batch_size != None:
            self.batch_size = batch_size

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data:np.ndarray, target_data:np.ndarray)->at.Matrix:
        # 前向传播，每条数据跑一个 forword

        # 调整数据格式
        input_data = gp.reshape_vector(input_data)
        input_data = at.MIdentityOp()(input_data)
        target_data = at.MIdentityOp()(target_data)
        output = None

        # 前向传播
        for layer in self.layers:
            output = layer.forward(input_data)
            input_data = output
            
        # 计算损失, loss = 1/batch_size * loss_function(output, target_data)
        self.curloss = self.loss_function(output, target_data)
        self.curloss.figMul(1 / self.batch_size)
        self.loss_log.append(self.curloss)

        return output

    def backward(self):
        # 根据不同的优化器去更新参数
        if self.optimizer == None:
            # 没有优化器，全跑一遍
            for loss in self.loss_log:
                tmp = at.Executor_Matrix(loss)
                tmp.gradients()
                for layer in self.layers[::-1]:
                    layer.backward()    
                    # 更新参数
                    layer.weight_Matrix.value -= self.learning_rate * layer.weight_Matrix.grad
                    layer.bias_Matrix.value -= self.learning_rate * layer.bias_Matrix.grad
                    
        elif self.optimizer == 'SGD':
            # SGD 优化
            # 随机抽取 1% 的样本数量的下标
            tmp_loss_list = [self.loss_log[randint(0, self.batch_size - 1)] for _ in range(self.batch_size // 100)]
            for loss in tmp_loss_list:
                loss = loss.figMul()
                tmp = at.Executor_Matrix(loss)
                tmp.gradients()
                for layer in self.layers[::-1]:
                    layer.backward()    
                    layer.weight_Matrix.value -= self.learning_rate * layer.weight_Matrix.grad
                    layer.bias_Matrix.value -= self.learning_rate * layer.bias_Matrix.grad
        
        elif self.optimizer == 'LGD':
            pass
        
        else:
            self.log.error('optimizer not found')
            raise ValueError('optimizer not found')
    
    def eval(self, input_data_list:np.ndarray, target_data_list:np.ndarray, isAccuracy = False):
        # 评估模型, 输入参数为测试集，记录 accuracy 和 loss
        losssaver, accsaver, acc_cnt, max_acc = Loss_Saver(), Loss_Saver(), 0, 0.0
        self.log.info('start evaluating!')
        self.batch_size = len(input_data_list)
        self.loss_log.clear()
        # for i, (input_data, target_data) in enumerate(zip(input_data_list, target_data_list)):
        for i in tqdm(range(len(input_data_list))):
            input_data = input_data_list[i]
            target_data = target_data_list[i]
            output_data = self.forward(input_data, target_data).value
            if isAccuracy:
                # 计算 accuracy
                acc_cnt += np.argmax(output_data) == np.argmax(target_data)
                acc = acc_cnt / (i + 1)
                max_acc = max(max_acc, acc)
                accsaver.updata(0, acc)
            losssaver.updata(0, self.curloss.value) # eval 时无 epoch 默认是 0
        # './NetWork/eval' + 日期命名
        losssaver.loss_saving(self.model_path + '/eval_loss.csv')
        accsaver.loss_saving(self.model_path + '/eval_acc.csv')
        self.log.info(f'max accuracy: {max_acc}')
        self.log.info('evaluating loss saved!')
        self.log.info('evaluating finished!')

    def train(self, input_data_list:pd.DataFrame, target_data_list:pd.DataFrame)->None:
        # 训练模型
        self.log.info(f'model path: {self.model_path} created!')

        losssaver = Loss_Saver()
        self.log.info('start training!')
        for e in range(self.epoch):
            self.epoch = e
            for i in tqdm(range(0, len(input_data_list), self.batch_size), desc='epoch_' + str(e)):
                # 清一下 loss 记录
                self.loss_log.clear()      
                self.batch_size = min(self.batch_size, len(input_data_list) - i)
                input_Matrixs = input_data_list[i: i + self.batch_size]
                target_Matrixs = target_data_list[i: i + self.batch_size]
                for input_data, target_data in zip(input_Matrixs, target_Matrixs):
                    self.forward(input_data, target_data)
                    losssaver.updata(e, self.curloss.value)
                self.backward()
            self.log.info(f'epoch {e} finished!')
        losssaver.loss_saving(self.model_path + '/train_loss.csv')
        # log 训练数据
        self.log.info('training loss saved!')
        self.log.info('training finished!')

    def save_model(self):
        path = self.model_path + '/model.json'
        '''
        connect1: {
            weight: [[], [], ...],
            bias: [],
            activate: 'ReLU',
            shape: (25, 25)
        },
        '''
        model_dict = {}
        for layer in self.layers:
            model_dict[layer.name + str(layer.id)] = {
                'weight': layer.weight_Matrix.value.tolist(),
                'bias': layer.bias_Matrix.value.tolist(),
                'activate': layer.activate if layer.activate != None else 'None',
                'shape': layer.weight_Matrix.shape
            }
        json.dump(model_dict, open(path, 'w'))
        self.log.info('model saved!')
        
    def load_model(self, path):
        # 加载模型
        model_dict = json.load(open(path, 'r'))
        for layer in model_dict.keys():
            layer_name = layer[:-1]
            # 根据layer_name创建layer
            if layer_name == 'Connected':
                pre_layer = Connected(shape=model_dict[layer]['shape'], weight=model_dict[layer]['weight'], bias=model_dict[layer]['bias'], activate=model_dict[layer]['activate'])
            elif layer_name == 'Convolution':
                pass
            elif layer_name == 'Pooling':
                pass
            else:
                raise ValueError('layer not found')
            self.layers.append(pre_layer)
        self.log.info('model loaded!')

if __name__ == '__main__':
    name = "Connected1"
    print(name[:-1])
