import numpy
import gzip
import struct
import matplotlib.pyplot as plt
from PIL import Image
import os
from pylab import *
#数据集读入代码：
def _read(image, label):
    mnist_dir = "data"# 存放数据集的文件夹
    with gzip.open(os.path.join(mnist_dir,label)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = numpy.fromstring(flbl.read(), dtype=numpy.int8)
    
    with gzip.open(os.path.join(mnist_dir,image), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = numpy.fromstring(fimg.read(), dtype=numpy.uint8).reshape(num, rows, cols)
        # 读入num张rows*cols大小的图片
    return image, label
def get_data():
    train_img, train_label = _read('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    test_img, test_label = _read('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    return train_img[:6000], train_label[:6000], test_img[:500], test_label[:500]
# train_img,train_label,test_img,test_label = get_data()
# figure()
# gray()
# for i in range(6000):
#     imshow(train_img[i])
#     show()