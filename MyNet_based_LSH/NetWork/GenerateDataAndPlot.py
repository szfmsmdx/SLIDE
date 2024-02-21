import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def reshape_vector(input_data):
    # 转化为 (1, x) 的矩阵
    if input_data.ndim < 3:
        return input_data.reshape(1, -1)
    return np.array([i.reshape(-1) for i in input_data])

def norm_data(data):
    # 归一化
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def GenerateClassifyData(num_of_input, input_shape, class_num):
    # 加 int 是因为输入科学计数法默认 float
    num_of_input = int(num_of_input)
    input_data = np.random.randn(num_of_input, *input_shape)
    output_data = np.zeros((num_of_input, class_num)).astype(float)
    for i in range(num_of_input):
        output_data[i][np.random.randint(0, class_num)] = 1
    # output_data = np.eye(class_num)[num_of_output]
    return input_data, output_data
    
def GenerateRegressionData(num_of_input, input_shape, output_shape):
    input_data = np.random.randn(num_of_input, *input_shape)
    output_data = np.random.randn(num_of_input, *output_shape)
    return input_data, output_data

def Generate_weight_bias(shape):
    weight = np.random.randn(*shape)
    bias = np.random.randn(shape[-1])
    return weight, bias

# plot Data --------------------------------------------------------------------------
# Plot csv file
def plot_csv(path, ylabel = 'loss', title = 'loss vs iteration'):
    df = pd.read_csv(path)
    # df['epoch'] = df['epoch'].astype(int)
    df['loss'] = df['loss'].round(3)
    # total_epochs = len(df['epoch'].unique())
    # epoch_len = int(len(df['epoch']) / total_epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['loss'], label='loss')
    # for i in range(total_epochs):
    #     ax.plot(range(i * epoch_len, (i + 1) * epoch_len), df['loss'][i * epoch_len:(i + 1) * epoch_len], color = 'C' + str(i))
    # percent_interval = max(1, int(total_epochs * epoch_len * 0.01))
    # xticks = np.arange(0, total_epochs * epoch_len, percent_interval)
    # ax.set_xticks(xticks)
    plt.xlabel('iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    path = r'C:\Users\szfmsmdx\Desktop\毕业论文\MyNet_based_LSH\NetWork\Model\2024-02-17_23：00：42\train_loss.csv'
    plot_csv(path)