"""
0401，黄圣杰，第一版本，可以简单存储ndarray类型数据到tfrecord
"""

import random
import numpy as np
import tensorflow as tf
from numpy.random import RandomState

# 统一用ndarray格式来保存，比较好（斌）
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
    X = rdm.rand(dataset_size, 2)
    # 定义输出值，将x1+x2 < 1的输入数据定义为正样本
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]  # 这个是list类型

    print(X)
    print(Y)

    Y_stacked = np.vstack(Y)
    print(Y_stacked.shape)  # (128, 1) 这个是ndarray类型

    print(X.shape)
    print(Y_stacked.shape)

    # 写入TFRecord文件
    output_file = './test0401.tfrecord'
    save_tfrecords2(X, Y_stacked, output_file)

    # 读取TFRecord文件并打印出其内容
    for example in tf.python_io.tf_record_iterator(output_file):
        print(tf.train.Example.FromString(example))

    # 读取TFRecord文件并还原成numpy array，再打印出来
    with tf.Session() as sess:
        dataset = tf.data.TFRecordDataset(output_file)  # 加载TFRecord文件
        dataset = dataset.map(parse_fn2)  # 解析data到Tensor
        dataset = dataset.repeat(1)  # 重复N epochs
        dataset = dataset.batch(3)  # batch size

        iterator = dataset.make_one_shot_iterator()
        next_data = iterator.get_next()

        while True:
            try:
                position, label = sess.run(next_data)
                print(position)
                print(label)
            except tf.errors.OutOfRangeError:
                break



















