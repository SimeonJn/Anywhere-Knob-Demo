import numpy as np

class MultiDimensionalRingBuffer:
    def __init__(self, size, dimensions):
        """
        初始化一个多维环形缓冲器。
        
        :param size: 缓冲器的大小
        :param dimensions: 数据的维度
        """
        self.size = size
        self.dimensions = dimensions
        self.buffer = [np.zeros(dimensions) for _ in range(size)]
        self.index = 0
        self.full = False

    def push(self, data):
        """
        向缓冲器中添加一个新的数据点，如果已满则覆盖旧的数据。
        
        :param data: 新的数据点，需要是一个与dimensions匹配的数组
        """
        if not self.full and self.index == self.size:
            self.full = True
        self.buffer[self.index % self.size] = data
        self.index += 1

    def reset(self):
        """
        重置缓冲器，清空所有数据。
        """
        self.buffer = [np.zeros(self.dimensions) for _ in range(self.size)]
        self.index = 0
        self.full = False

    def get_avg(self):
        """
        计算并返回缓冲器中所有数据点的平均值。
        
        :return: 平均值向量
        """
        if self.index == 0:
            return np.zeros(self.dimensions)
        elif self.full:
            return np.mean(self.buffer, axis=0)
        else:
            return np.mean(self.buffer[:self.index], axis=0)

    def get_sigma(self):
        """
        计算并返回缓冲器中所有数据点的标准差。
        
        :return: 标准差向量
        """
        avg = self.get_avg()
        variance = np.zeros(self.dimensions)
        count = min(self.index, self.size)
        for data in self.buffer[:count]:
            variance += (data - avg)**2
        return np.sqrt(variance / count) if count > 1 else np.zeros(self.dimensions)
    
    def get_sigma_avg(self):
        """
        计算并返回缓冲器中所有数据点的标准差和平均值。
        
        :return: 标准差和平均值向量
        """
        sigma = self.get_sigma()
        return sigma.sum() / 2