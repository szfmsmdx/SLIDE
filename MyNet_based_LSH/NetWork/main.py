import warnings
warnings.filterwarnings("ignore")
import GenerateDataAndPlot as gp
from layer import Connected
from Net import Net


if __name__ == '__main__':
    # 流程

    # ---------------------------------------------------------------------------
    # 生成数据
    INPUT_SHAPE = (25, 25)
    NETWORK_SHAPE = [4]
    OUTPUT_SHAPE = 3
    input_data, output_data = gp.GenerateClassifyData(10, INPUT_SHAPE, OUTPUT_SHAPE)
    print(input_data.shape, output_data.shape)

    # ---------------------------------------------------------------------------
    # 调整数据格式
    # input_data = gp.reshape_vector(input_data)
    # input_data = at.MIdentityOp()(input_data)

    # ---------------------------------------------------------------------------
    # 初始化网络
    # layer1 = Connected((input_data.shape[1], NETWORK_SHAPE[0]), 'Sigmoid')
    # layer2 = Connected((NETWORK_SHAPE[0], NETWORK_SHAPE[1]), 'ReLU')
    # layer3 = Connected((NETWORK_SHAPE[1], NETWORK_SHAPE[2]), 'Tanh')
    # layer4 = Connected((NETWORK_SHAPE[2], output_data.shape[1]), 'Softmax')
    # output = layer1.forward(input_data)
    # output = layer2.forward(output)
    # output = layer3.forward(output)
    # output = layer4.forward(output)
    # layer4.backward()
    # layer3.backward()
    # layer2.backward()
    # layer1.backward()
    # print(input_data.grad)
    # print(output.value)

    Net = Net()
    Net.add(Connected(shape=(INPUT_SHAPE[0] * INPUT_SHAPE[1], NETWORK_SHAPE[0]), activate='ReLU', weight=None, bias=None))
    Net.add(Connected(shape=(NETWORK_SHAPE[0], OUTPUT_SHAPE), activate='Softmax', weight=None, bias=None))

    # ---------------------------------------------------------------------------
    # 训练网络
    Net.set(learning_rate=0.2, loss_function='MSE', epoch=1, batch_size=1)
    Net.train(input_data, output_data)

    # ---------------------------------------------------------------------------
    # 测试网络
    
    # ---------------------------------------------------------------------------
    # 画图

    # ---------------------------------------------------------------------------
    # 保存网络
    Net.save_model()
    # ---------------------------------------------------------------------------
    # 读取网络
