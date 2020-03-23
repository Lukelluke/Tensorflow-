# https://blog.csdn.net/junjun150013652/article/details/81282967
import tensorflow as tf
#输入为100个数据集，28*28像素，3个通道
input = tf.Variable(tf.random_normal([100,28,28,3]))
#卷积核过滤器尺寸为3*3，输入3个通道，输出10个通道，也就是10个卷积核
filter = tf.Variable(tf.random_normal([3,3,3,10]))
 
output = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='VALID')

print(output)

"""
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
"""
"""
输入参数：

input: 一个四维的Tensor. 必须是以下类型中的一种: 
half, bfloat16, float32, float64. 
维度顺序根据data_format的值进行解释，详细信息请参见下文。

filter:  必须和input有相同的类型. 一个四维的Tensor，
shape=[filter_height, filter_width, in_channels, out_channels]，
这里的in_channels就是input中的in_channels，
out_channels就是卷积核的个数

strides: 长度为4的一维tensor，类型是整数，每维的滑动窗口的步幅。
维度顺序由data_format的值确定，有关详细信息，请参见下文 

padding: 一个字符串: "SAME", "VALID". "SAME"时，
输出宽或高 = ceil(输入宽或高/步幅)，"VALID"时，
输出宽或高 = ceil((输入宽或高-filter的宽或高+1)/步幅)，
ceil() 为向上取整。

use_cudnn_on_gpu: 一个可选的 bool类型.默认是True.

data_format: 来自“NHWC”，“NCHW”的可选字符串，
默认为“NHWC”，指定输入和输出数据的数据格式。
使用默认格式“NHWC”，数据按以下顺序存储：
[batch, height, width, channels]  
或者，格式可以是“NCHW”，数据存储顺序为：
[batch, channels, height, width]

返回值：一个和 input有相同类型的Tensor.
"""