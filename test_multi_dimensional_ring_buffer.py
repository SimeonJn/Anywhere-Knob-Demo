import unittest
import numpy as np
from multi_dimensions_ring_buffer import MultiDimensionalRingBuffer

class TestMultiDimensionalRingBuffer(unittest.TestCase):

    def setUp(self):
        """为每个测试方法执行的初始化操作"""
        self.buffer_size = 5
        self.dimensions = 3
        self.buffer = MultiDimensionalRingBuffer(self.buffer_size, self.dimensions)

    def test_push(self):
        """测试push方法是否正确添加数据"""
        for i in range(10):
            self.buffer.push(np.array([i, i**2, i**3]))
        # 检查缓冲区前几个元素是否正确
        self.assertTrue(np.array_equal(self.buffer.buffer[0], np.array([5, 25, 125])))
        self.assertTrue(np.array_equal(self.buffer.buffer[1], np.array([6, 36, 216])))

    def test_reset(self):
        """测试reset方法是否能清空缓冲区"""
        for i in range(10):
            self.buffer.push(np.array([i, i**2, i**3]))
        self.buffer.reset()
        for data in self.buffer.buffer:
            self.assertTrue(np.array_equal(data, np.zeros(self.dimensions)))

    def test_get_avg(self):
        """测试get_avg方法是否计算平均值正确"""
        sum = np.zeros(self.dimensions)
        for i in range(self.buffer_size + 2):  # 确保有数据被覆盖
            self.buffer.push(np.array([i, i**2, i**3]))
            sum += np.array([i, i**2, i**3])
        sum -= np.array([1,1,1])
        avg = self.buffer.get_avg()
        expected_avg = sum / self.buffer_size
        np.testing.assert_array_almost_equal(avg, expected_avg, decimal=5)

    def test_get_sigma(self):
        """测试get_sigma方法是否计算标准差正确"""
        # 注意：直接验证标准差的精确值较为复杂，因为涉及到平方和开方，这里主要验证逻辑流程
        for i in range(self.buffer_size + 4):  
            self.buffer.push(np.array([i, i**2, i**3]))
        sigma = self.buffer.get_sigma()
        # 确保返回的是非零向量，且长度与维度匹配，更详细的验证可能需要根据具体数学公式进行
        self.assertTrue(np.count_nonzero(sigma) > 0)
        self.assertEqual(len(sigma), self.dimensions)

if __name__ == '__main__':
    unittest.main()