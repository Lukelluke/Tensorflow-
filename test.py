"""
有个小疑问，就是送进tfrecord的label尺寸是（128，1）
怎么出来就变成（batch_size，）了，少了一维？

"""

import random
import numpy as np
import tensorflow as tf
from numpy.random import RandomState


def save_tfrecords2(position_data, label_data, dest_file):
    with tf.python_io.TFRecordWriter(dest_file) as writer:
        for i in range(len(position_data)):
            features = tf.train.Features(
                feature={
                    "position": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[position_data[i].astype(np.float64).tostring()])),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label_data[i]]))
                }
            )
            tf_example = tf.train.Example(features=features)
            serialized = tf_example.SerializeToString()
            writer.write(serialized)


def parse_fn2(example_proto):
    features = {"position": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return tf.decode_raw(parsed_features['position'], tf.float64), parsed_features['label']


if __name__ == '__main__':
    # buffer_s, buffer_a = [], []

    rdm = RandomState(1)
    # 定义数据集的大小
    dataset_size = 128
    # 模拟输入是一个二维数组
    X = rdm.rand(dataset_size, 2)  # 得到一个（128，2）大小的数组，类型是ndarray
    # ******这个是例子里面的分类方案；*******
    # 定义输出值，将x1+x2 < 1的输入数据定义为正样本
    #Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]  # 这个是list类型
    Y = [[int(x1 < 0)] for (x1, x2) in X]  # 这里咱们重新定义为简单判断正负半轴

    Y_stacked = np.vstack(Y)  # 将list类型转化为 ndarray 类型

    print("**********************")
    print(Y_stacked.shape)  # (128, 1) 这个是ndarray类型
    print(type(Y_stacked))  # ndarray
    print(X.shape)  # (128, 2)

    # 写入TFRecord文件
    output_file = './test0401.tfrecord'
    save_tfrecords2(X, Y_stacked, output_file)

    # 以下是关于网络搭建：
    batch_size = 8
    # 定义神经网络的参数
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
    # 定义输入和输出
    x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
    # 定义神经网络的前向传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 定义损失函数和反向传播算法
    # 使用交叉熵作为损失函数
    # tf.clip_by_value(t, clip_value_min, clip_value_max,name=None)
    # 基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况

    # cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))  # 这事例子里面采用的

    cross_entropy = tf.losses.mean_squared_error(y_, y)  # 咱们先简单采用"MSE"均方误差来处理


    # 使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    ## *************数据已经保存到具体的tfrecord当中去了，等下调用：
    # 通过随机函数生成一个模拟数据集
    # rdm = RandomState(1)
    # # 定义数据集的大小
    # dataset_size = 128
    # # 模拟输入是一个二维数组:rand(行数，列数)
    # X = rdm.rand(dataset_size, 2)
    # # 定义输出值，将x1+x2 < 1的输入数据定义为正样本
    # Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

    init = tf.initialize_all_variables()

    # 读取TFRecord文件并还原成numpy array，再打印出来
    dataset = tf.data.TFRecordDataset(output_file)  # 加载TFRecord文件
    # print("line64: dataset.type = "+str(type(dataset)))
    # dataset.type = <class 'tensorflow.python.data.ops.readers.TFRecordDataset'>

    dataset = dataset.map(parse_fn2)  # 解析data到Tensor  （data.map()相当于把data里面的数据，全部都进行一次map里面的函数运算）
    dataset = dataset.repeat(1)  # 重复N epochs
    dataset = dataset.batch(batch_size)  # batch size

    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    next_data = iterator.get_next()

    with tf.Session() as sess:

        sess.run(init)
        steps = 10000

        sess.run(iterator.initializer)

        for i in range(steps):
            try:
                position, label = sess.run(next_data)  # 都是numpy.ndarray类型
                label = np.reshape(label, (batch_size, 1))
                # print(position)
                # print(label)  # [0 1 0 1 0 1 0 0]
                # print("label.shape = " + str(label.shape))  # label.shape = (8,1)

                sess.run(train_step, feed_dict={x: position, y_: label})
                # 每迭代1000次输出一次日志信息
                if i % 1000 == 0:
                    # 计算所有数据的交叉熵
                    total_cross_entropy = sess.run(cross_entropy, feed_dict={x: position, y_: label})
                    # 输出交叉熵之和
                    print("After %d training step(s),cross entropy on all data is %.12f" % (i, total_cross_entropy))



            except tf.errors.OutOfRangeError:
                # print("i= "+str(i))
                if i == steps:
                    print("************************************")
                    break
                else:
                    # print("hi,I'm here")
                    sess.run(iterator.initializer)
        # 输出参数w1
        print(w1.eval(session=sess))
        # 输出参数w2
        print(w2.eval(session=sess))





















