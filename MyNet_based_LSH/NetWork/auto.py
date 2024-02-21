# -* encoding:utf-8 *-
import math
import numpy as np

class Node(object):
  """
  表示具体的数值或者某个Op的数据结果。
  """
  global_id = -1
  
  def __init__(self, op, inputs):
    self.inputs = inputs # 产生该Node的输入
    self.op = op # 产生该Node的Op
    self.grad = 0.0 # 初始化梯度
    self.evaluate() # 立即求值
    # 调试信息
    self.id = Node.global_id
    Node.global_id += 1
    print("eager exec: %s" %  self)
  
  def input2values(self):
    """ 将输入统一转换成数值，因为具体的计算只能发生在数值上 """
    new_inputs = []
    for i in self.inputs:
      if isinstance(i, Node):
        i = i.value
      new_inputs.append(i)
    return new_inputs

  def evaluate(self):
    self.value = self.op.compute(self.input2values())

  # def __repr__(self):
  #   return self.__str__()

  # def __str__(self):
  #   return "Node%d: %s %s = %s, grad: %.3f" % (
  #     self.id, self.input2values(), self.op.name(), self.value, self.grad
  #     )

class Op(object):
  """
  所有操作的基类。注意Op本身不包含状态，计算的状态保存在Node中，每次调用Op都会产生一个Node。
  """
  def name(self):
    pass
  
  def __call__(self):
    """ 产生一个新的Node，表示此次计算的结果 """
    pass

  def compute(self, inputs):
    """ Op的计算 """
    pass

  def gradient(self, output_grad):
    """ 计算梯度 """
    pass

class AddOp(Op): 
  """加法运算"""
  def name(self):
    return "add"

  def __call__(self, a, b):
    return Node(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] + inputs[1]
  
  def gradient(self, inputs, output_grad):
    return [output_grad, output_grad] # gradient of a and b

class SubOp(Op): 
  """减法运算"""
  def name(self):
    return "sub"

  def __call__(self, a, b):
    return Node(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] - inputs[1]
  
  def gradient(self, inputs, output_grad):
    return [output_grad, -output_grad]

class MulOp(Op): 
  """乘法运算"""
  def name(self):
    return "mul"
  
  def __call__(self, a, b):
    return Node(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] * inputs[1]

  def gradient(self, inputs, output_grad):
    return [inputs[1] * output_grad, inputs[0] * output_grad]

class LnOp(Op): 
  """自然对数运算"""
  def name(self):
    return "ln"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.log(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [1.0/inputs[0] * output_grad]

class SinOp(Op): 
  """正弦运算"""
  def name(self):
    return "sin"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.sin(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [math.cos(inputs[0]) * output_grad]

class IdentityOp(Op): 
  """输入输出一样"""
  def name(self):
    return "identity"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return inputs[0]
  
  def gradient(self, inputs, output_grad):
    return [output_grad]

class ExpOp(Op):
    """指数运算"""
    def name(self):
        return "exp"
    
    def __call__(self, a):
        return Node(self, [a])
    
    def compute(self, inputs):
        return math.exp(inputs[0])
    
    def gradient(self, inputs, output_grad):
        return [math.exp(inputs[0]) * output_grad]

class Executor(object):
  """ 计算图的执行和自动微分 """

  def __init__(self, root):
    self.topo_list = self.__topological_sorting(root) # 拓扑排序的顺序就是正向求值的顺序
    self.root = root

  def run(self):
    """
    按照拓扑排序的顺序对计算图求值。注意：因为我们之前对node采用了eager模式，
    实际上每个node值之前已经计算好了，但为了演示lazy计算的效果，这里使用拓扑
    排序又计算了一遍。
    """
    node_evaluated = set() # 保证每个node只被求值一次
    print("\nEVALUATE ORDER:")
    for n in self.topo_list:
      if n not in node_evaluated:
        n.evaluate()
        node_evaluated.add(n)
        print("evaluate: %s" % n)
    
    return self.root.value

  def __dfs(self, topo_list, node):
    if Node == None or not isinstance(node, Node):
      return
    for n in node.inputs:
      self.__dfs(topo_list, n)
    topo_list.append(node) # 同一个节点可以添加多次，他们的梯度会累加

  def __topological_sorting(self, root):
    """拓扑排序：采用DFS方式"""
    lst = []
    self.__dfs(lst, root)
    return lst

  def gradients(self):
    reverse_topo = list(reversed(self.topo_list)) # 按照拓扑排序的反向开始微分
    reverse_topo[0].grad = 1.0 # 输出节点梯度是1.0
    for n in reverse_topo:
      grad = n.op.gradient(n.input2values(), n.grad)
      # 将梯度累加到每一个输入变量的梯度上
      for i, g in zip(n.inputs, grad):
        if isinstance(i, Node):
          i.grad += g
    print("\nAFTER AUTODIFF:")
    for n in reverse_topo:
      print(n)

# ------------------------------------------------------------------
# 以下是矩阵运算
# 传入Matrix，输出Matrix
# MIdentityOp 可以接受 np.array
# MAddOp, DotOp, ReLUOp, SigmoidOp, SoftmaxOp, TanhOp, CrossEntropyOp, MSEOp 只能接受 Matrix
class Matrix:
  global_id = -1

  def __init__(self, op, inputs):
    self.inputs = np.array(inputs) # 产生该Matrix的输入
    self.op = op
    self.evaluate() # 立即求值
    self.shape = self.value.shape
    self.grad = np.zeros(self.shape) # 初始化梯度
    self.T = self.value.T
    self.T_shape = self.T.shape
    self.T_grad = np.zeros(self.T_shape)

    # 调试信息
    self.id = Matrix.global_id
    Matrix.global_id += 1

  def input2values(self):
    """ 将输入统一转换成数值，因为具体的计算只能发生在数值上 """
    new_inputs = []
    for i in self.inputs:
      if isinstance(i, Matrix):
        i = i.value
      new_inputs.append(i)
    return new_inputs

  def evaluate(self):
    self.value = self.op.compute(self.input2values())
    
  def figMul(self, k):
    '''矩阵乘以一个常数'''
    self.value = self.value * k
    self.grad = self.grad * k
    self.T = self.T * k
    self.T_grad = self.T_grad * k
    
class MOp(object):
  '''
  矩阵操作的基类
  '''
  def name(self):
        pass
  
  def __call__(self):
    pass

  def compute(self, inputs):
    pass
  
  def gradient(self, inputs, output_grad):
    pass
  
class NormOp(MOp):
  """归一化运算"""
  def name(self):
    return "norm"

  def __call__(self, a):
    return Matrix(self, [a])

  def compute(self, inputs):
    if np.max(inputs[0]) == np.min(inputs[0]):
      return inputs[0]
    else:
      return (inputs[0] - np.min(inputs[0])) / (np.max(inputs[0]) - np.min(inputs[0]))
  
  def gradient(self, inputs, output_grad):
    return [output_grad]

class MAddOp(MOp):
  """加法运算"""
  def name(self):
    return "add"

  def __call__(self, a, b):
    return Matrix(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] + inputs[1]
  
  def gradient(self, inputs, output_grad):
    return [output_grad, output_grad] # gradient of a and b
  
class DotOp(MOp):
  """矩阵乘法运算"""
  def name(self):
    return "dot"

  def __call__(self, a, b):
    return Matrix(self, [a, b])

  def compute(self, inputs):
    return inputs[0].dot(inputs[1])

  def gradient(self, inputs, output_grad):
    return [output_grad.dot(inputs[1].T), inputs[0].T.dot(output_grad)]

class ReLUOp(MOp):
  """ReLU运算"""
  def name(self):
    return "relu"

  def __call__(self, a):
    return Matrix(self, [a])

  def compute(self, inputs):
    return np.maximum(inputs[0], 0)

  def gradient(self, inputs, output_grad):
    return [np.where(inputs[0] > 0, output_grad, 0)]

class SigmoidOp(MOp):
  """Sigmoid运算"""
  def name(self):
    return "sigmoid"

  def __call__(self, a):
    return Matrix(self, [a])

  def compute(self, inputs):
    return 1.0 / (1.0 + np.exp(-inputs[0]))

  def gradient(self, inputs, output_grad):
    return [output_grad * self.compute(inputs) * (1.0 - self.compute(inputs))]
  
class MIdentityOp(MOp):
    """输入输出一样"""
    def name(self):
        return "identity"
    
    def __call__(self, a):
        return Matrix(self, [a])
    
    def compute(self, inputs):
        return inputs[0]
    
    def gradient(self, inputs, output_grad):
        return [output_grad]

class SoftmaxOp(MOp):
    """Softmax运算"""
    def name(self):
        return "softmax"
    
    def __call__(self, a):
        return Matrix(self, [a])
    
    def compute(self, inputs):
        return np.exp(inputs[0]) / np.sum(np.exp(inputs[0]), axis=1, keepdims=True)
    
    def gradient(self, inputs, output_grad):
        return [output_grad * self.compute(inputs) * (1.0 - self.compute(inputs))]

class TanhOp(MOp):
    """Tanh运算"""
    def name(self):
        return "tanh"
    
    def __call__(self, a):
        return Matrix(self, [a])
    
    def compute(self, inputs):
        return np.tanh(inputs[0])
    
    def gradient(self, inputs, output_grad):
        return [output_grad * (1.0 - np.square(self.compute(inputs)))]

class CrossEntropyOp(MOp):
    """交叉熵运算"""
    def name(self):
        return "cross_entropy"
    
    def __call__(self, a, b):
        return Matrix(self, [a, b])
    
    def compute(self, inputs):
        return -np.sum(inputs[1] * np.log(inputs[0]))
    
    def gradient(self, inputs, output_grad):
        return [output_grad * (-inputs[1] / inputs[0])]
    
class MSEOp(MOp):
    """均方误差运算"""
    def name(self):
        return "mse"
    
    def __call__(self, a, b):
        return Matrix(self, [a, b])
    
    def compute(self, inputs):
        return np.sum(np.square(inputs[0] - inputs[1]))
    
    def gradient(self, inputs, output_grad):
        return [output_grad * 2 * (inputs[0] - inputs[1])]

class Executor_Matrix(object):
  '''
  矩阵形式的计算图的执行和自动微分
  '''
  def __init__(self, root):
    self.topo_list = self.__topological_sorting(root)
    self.root = root

  def run(self):
    matrix_evaluated = set()
    for n in self.topo_list:
      if n not in matrix_evaluated:
        n.evaluate()
        matrix_evaluated.add(n)
    return self.root.value
  
  def __dfs(self, topo_list, matrix):
    if Matrix == None or not isinstance(matrix, Matrix):
      return
    for n in matrix.inputs:
      self.__dfs(topo_list, n)
    topo_list.append(matrix) # 同一个矩阵可以添加多次，他们的梯度会累加

  def __topological_sorting(self, root):
    """拓扑排序：采用DFS方式"""
    lst = []
    self.__dfs(lst, root)
    return lst

  def gradients(self):
    reverse_topo = list(reversed(self.topo_list))
    reverse_topo[0].grad = np.ones(reverse_topo[0].value.shape)
    for n in reverse_topo:
      grad = n.op.gradient(n.input2values(), n.grad)
      # 将梯度累加到每一个输入变量的梯度上
      for i, g in zip(n.inputs, grad):
        if isinstance(i, Matrix):
          i.grad += g


# -------------------------------------------
# 开始验证程序
# add, mul, ln, sin, sub, identity, exp = AddOp(), MulOp(), LnOp(), SinOp(), SubOp(), IdentityOp(), ExpOp()
# x1, x2 = identity(2.0), identity(5.0)
# y = sub(add(ln(x1), mul(x1, x2)), sin(x2)) # y = ln(x1) + x1*x2 - sin(x2)
# ex = Executor(y)
# print("y=%.3f" % ex.run())
# ex.gradients() # 反向计算 自动微分
# print("x1.grad=%.3f" % x1.grad)
# print("x2.grad=%.3f" % x2.grad)

if __name__ == '__main__':
  add, dot, relu, sigmoid, softmax = MAddOp(), DotOp(), ReLUOp(), SigmoidOp(), SoftmaxOp()
  a = np.random.randn(3, 3)
  b = np.random.randn(3, 3)
  a = MIdentityOp()(a)
  b = MIdentityOp()(b)
  c = MSEOp()(b, a)
  c.figMul(1)
  print(c)
  ex = Executor_Matrix(c)
  ex.gradients()
  print(ex)
  print(a.grad)
  print(b.grad)
