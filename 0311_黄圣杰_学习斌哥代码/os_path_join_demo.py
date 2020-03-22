import os 
# os 标准库https://www.runoob.com/python3/python3-os-file-methods.html

# os.path.join()函数：连接两个或更多的路径名组件

# 1.如果各组件名首字母不包含’/’，则函数会自动加上
# 2.如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
# 3.如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾


path1 = 'home'
path2 = 'develop'
path3 = 'code'

path10 = path1 + path2 + path3
path20 = os.path.join(path1,path2,path3)

print('path10 = ',path10)
print('path20 = ',path20)


print('\n*************************\n')

path1 = 'home'
path2 = 'develop'
path3 = '/code'

path10 = path1 + path2 + path3
path20 = os.path.join(path1,path2,path3)

print('path10 = ',path10)
print('path20 = ',path20)













