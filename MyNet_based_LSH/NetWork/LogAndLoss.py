import logging
import numpy as np
import pandas as pd

root_nowexp = "test"

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Loss_Saver:
    def __init__(self, moving=False):
        self.epoch_list, self.loss_list, self.last_loss =[], [], 0.0
        self.moving = moving  # 是否进行滑动平均操作
	
    def updata(self, epoch, value):
		# 只有进行滑动平均时，才会用到 self.last_loss 
        self.epoch_list += [epoch]
        if not self.moving:
            self.loss_list += [value]
        elif not self.loss_list:
            self.loss_list += [value]
            self.last_loss = value
        else:
            update_val = self.last_loss * 0.9 + value * 0.1
            self.loss_list += [[update_val]]
            self.last_loss = update_val
        return

    def loss_saving(self, root_file, encoding='utf-8'):
    	# 这个用于在指定位置保存 loss 指标
        loss_array = np.array(self.loss_list)
        epoch_array = np.array(self.epoch_list)
        colname = ['epoch', 'loss']
        listPF = pd.DataFrame(columns=colname, data=np.column_stack((epoch_array, loss_array)))
        listPF.to_csv(root_file, encoding=encoding)
