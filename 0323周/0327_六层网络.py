import tensorflow as tf
import numpy as np
import librosa
import os, sys

# Hyperparameters
batch_size = 16
learning_rate = 0.01

# 随机取一秒钟： 先等下写
def Get_time(wav):
    pass

# 打开文件
path = "./wav/"
dirs = os.listdir(path)

a = np.zeros(shape=(16, 100, 80))  # 这个用来保存最终的（16，100，80）
count = 0
# 输出所有文件和文件夹
for file in dirs:
    wav, sr = librosa.load(os.path.join('./wav', file), sr=None)
    print("sample_rate = " + str(sr))  # sample_rate = 24000
    print("wav.shape = " + str(wav.shape))  # (75248,)
    new_mel = librosa.feature.melspectrogram(y=wav[24001:48000], sr=sr, n_mels=80, hop_length=240)  # n_mels 默认128,改为80维
    input_mel = new_mel.T  # （100，80）ndarray类型；
    print("new_mel.shape = " + str(new_mel.shape))  # new_mel.shape = (80, 100)
    print("input_mel.shape = " + str(input_mel.shape))  # input_mel.shape = (100, 80)，100帧，每帧的mel特征：80维度

    reshaped_input_mel = np.reshape(input_mel, (1, 100, 80))
    a[count] = reshaped_input_mel  # 特别笨的方法，主要是batch不知道怎么弄！
    count += 1

# a = (16,100,80)
# ori_mel = (16,100,80,1)

ori_mel = np.reshape(a, (16, 100, 80, 1))  # 用来送进网络的特征，16个音频

# print(ori_mel.shape)  # (16, 100, 80, 1)
# print(ori_mel)
# print("here line 37")
# print(count)
# print(a)

tf_x = tf.placeholder(tf.float32, [16, 100, 80, 1])  # 输入的placeholder


# Network :三层卷积（1，24，48，1） &  三层全连接（）
# input = (16 * 100 * 80 * 1)
# 最后一个参数是：通道，通过卷积变化的是通道数；
def Conv(input):
    con1 = tf.layers.conv2d(input, filters=24, kernel_size=3,
                            strides=1, padding="SAME", activation=tf.nn.relu)
    con2 = tf.layers.conv2d(con1, filters=48, kernel_size=3,
                            strides=1, padding="SAME", activation=tf.nn.relu)
    con3 = tf.layers.conv2d(con2, filters=1, kernel_size=3,
                            strides=1, padding="SAME", activation=tf.nn.relu)
    return con3


# # reshape 直接放到Dense里面做
# # Reshape,从(16 * 100 * 80 * 1)变成（16 * 100 * 80）
# # 因为全连接网络里面不需要通道，只有卷积才需要；
def Reshape(input):
    print("line69:input.shape = "+str(input.shape))
    out = tf.reshape(input, (16, 100, 80))
    print("line71:out.shape = "+str(out.shape))
    return out


# 全连接的cell个数，此时貌似不重要？
# 输入:（16 * 100 * 80 * 1），输出：（16 * 100 * 80）
def Dense(input):
    print("line 77:input.shape = "+str(input.shape))
    de1 = tf.layers.dense(input, 512, tf.nn.tanh)
    de2 = tf.layers.dense(de1, 512, tf.nn.tanh)
    de3 = tf.layers.dense(de2, 80, tf.nn.tanh)
    return de3


conved = Conv(tf_x)  # 卷积过后产物，tf_x = [16, 100, 80， 1]
print("line81:conved.shape = "+str(conved.shape))
reshaped = Reshape(conved)
print("line83:reshaped.shape = "+str(reshaped.shape))
densed = Dense(reshaped)  # 全连接后的产物 [16,100,80]

loss = tf.losses.mean_squared_error(labels=a, predictions=densed)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        _, loss_ = sess.run([train, loss], feed_dict={tf_x: ori_mel})

        if step % 100 == 0:
            print('step:{}，train loss = {:.4f}'.format(step, loss_))

        """
        这里需要提供数据来源
        之后就可以指定sess.run([loss, train],feed_dict={...})
        """

