import GenerateDataAndPlot as gp
import numpy as np
import auto as at

class layer:
    '''
    基类，包含前向传播和反向传播两个基本的方法
    '''
    def __init__(self) -> None:
        self.name = None
        pass
    def forward(self):
        pass
    def backward(self):
        pass

class Connected(layer):
    global_id = 0
    def __init__(self, shape, weight = None, bias = None, activate = 'ReLU') -> None:
        self.name = 'Connected'
        self.id = Connected.global_id
        Connected.global_id += 1

        self.shape = shape
        self.weight = weight if weight else np.random.randn(*shape)
        self.bias = bias if bias else np.random.randn(1, shape[1])
        self.weight_Matrix = at.MIdentityOp()(self.weight)
        self.bias_Matrix = at.MIdentityOp()(self.bias)
        # print(self.weight_Matrix.id, self.bias_Matrix.id,"\n")
        self.input_Matrix = None
        self.output_Matrix = None
        self.activate = activate

    def forward(self, input_Matrix):
        self.input_Matrix = input_Matrix
        self.before_activate = at.MAddOp()(at.DotOp()(self.input_Matrix, self.weight_Matrix), self.bias_Matrix)
        if self.activate == 'ReLU':
            self.output_Matrix = at.ReLUOp()(self.before_activate)
        elif self.activate == 'Sigmoid':
            self.output_Matrix = at.SigmoidOp()(self.before_activate)
        elif self.activate == 'Tanh':
            self.output_Matrix = at.TanhOp()(self.before_activate)
        elif self.activate == 'Softmax':
            # 先归一化
            self.before_activate = at.NormOp()(self.before_activate)
            self.output_Matrix = at.SoftmaxOp()(self.before_activate)
        else:
            self.output_Matrix = self.before_activate
        self.ex = at.Executor_Matrix(self.output_Matrix)
        return self.output_Matrix
    
    def backward(self):
        self.ex.gradients()  
        return self.input_Matrix

class Convolution(layer):
    # 卷积层
    pass

class Pooling(layer):
    # 池化层
    pass