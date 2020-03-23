# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# print(mnist.train.images.shape, mnist.train.lables.shape)  # 训练集
# print(mnist.test.images.shape, mnist.test.lables.shap)    # 测试集
# print(mnist.validation.images.shape, mnist.validation.lables.shape)  # 验证集

from tensorflow.examples.tutorials.mnist import input_data

print('packs loaded')
print('download and extract mnist set')
mnist = input_data.read_data_sets('MNIST_data/',
                                  one_hot=True)  # one_hot 是0,1的编码格式,read_data_sets 用于下载数据集(文件目录里有就不用下载,改成直接读取,没有就需要下载;

print('the type of mnist:%s' % type(mnist))
print('the number of train data:%d' % (mnist.train.num_examples))
print('the number of test data:%d' % (mnist.test.num_examples))

print('把数据集分开,分别有四个:')
print('1:train images,2:train labels,3:test images,4:test labels')
print('what does the mnist look like?')

#image图片，训练数据
train_img = mnist.train.images

#训练标签，
train_label = mnist.train.labels

#测试数据
test_img = mnist.test.images

#测试标签
test_label = mnist.test.labels
print(mnist.train.labels.shape)
print('type-------------->>>>>')
print('train images type%s'%type(train_img))
print('train labels type%s'%type(train_label))
print('test images type%s'%type(test_img))
print('test labels type%s'%type(test_label))
print('shape------------------->>>')
print('train images shape%s'%(train_img.shape,)) #特别要注意这个逗号,不然会报TypeError: not all arguments converted during string formatting
print('train labels shape%s'%(train_label.shape,))
print('test images shape%s'%(test_img.shape,))
print('test labels shape%s'%(test_label.shape,))