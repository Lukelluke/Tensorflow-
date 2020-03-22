a=1111111111111111111111
b=1111111111111111111111
print(id(a))
print(id(b))
print(a is b)

"""
python中没有++和--的，那么要实现自增和自减的话，可以使用如下操作：

a = a + 1 或 a += 1

原因分析

python中的数字类型是不可变数据，也就是数字类型数据在内存中不会发生改变，当变量值发生改变时，会新申请一块内存赋值为新值，然后将变量指向新的内存地址
a = 10; a += 1
两次id(a)是不同的

+=是改变变量，相当于重新生成一个变量，把操作后的结果赋予这个新生成的变量
--是改变了对象本身，而不是变量本身，即改变数据地址所指向的内存中的内容
————————————————
版权声明：本文为CSDN博主「lalalalalalaaaa」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/luoyehuixuanaaaa/article/details/94973082

但：

有时候在 Python 中看到存在 ++i 这种形式，这其实不是自增，
只是简单的表示正负数的正号而已。正正得正，负负得正，所以 ++i 和 --i 都是 i 。
https://blog.csdn.net/weixin_33883178/article/details/91447334?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
"""