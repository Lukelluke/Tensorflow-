# 分别用librosa、pydub、scipy.io、soundfile中的函数读取声音文件。
# 看一看它们的输出样式都是什么样的。
# short 为 int16类型
# https://blog.csdn.net/weixin_43695834/article/details/84995323
# 维度的缺省值，表示通道，2表示双通道，单通道就直接缺省
# 所以 f0 特征的维度应该是一维！！！

import numpy as np
import librosa
from pydub import AudioSegment
from scipy.io import wavfile
import soundfile as sf 
import pyworld as pw
# 贾宇康提到过的一种提取音频信息的库：soundfile

path = 'E10001.wav'



## librosa
y, sr = librosa.load(path, sr=None, mono=False,dtype=np.float64)  # sr=None声音保持原采样频率， 
                                                 # mono=False声音保持原通道数:是否将声音转为单声道
print('librosa.load 声音数据的维度，数据类型，最大值，中间值，最小值为：', 
	y.shape, y.dtype, np.max(y), np.median(y), np.min(y))



## pyworld
## f0是一维数组，每帧会有一个f0
f0,timeaxis = pw.harvest(y, sr)

print('f0 声音数据的维度，数据类型，最大值，中间值，最小值为：', 
	f0.shape, f0.dtype, np.max(f0), np.median(f0), np.min(f0))

print('\n*****\n')
print(sum(f0))
print('\n*****\n')

print(sum(timeaxis))
print(len(f0))

print(sum(f0)/len(f0))

def get_mean_f0(fpath:path):  # 求原始/目标人的均值，然后再拿出去做差
    wav, _ = librosa.load(fpath, sr, mono=True, dtype=np.float64)  # librosa.load 返回音频信号值 & 采样率
    f0, timeaxis = pw.harvest(wav, sr)  # f0是一维数组，每帧会有一个f0

    total_f0 = 0
    for i in range(int(sum(timeaxis)/0.005)):
        total_f0 += f0

    average_f0 = total_f0/(int(sum(timeaxis)/0.005))
    return average_f0
#print(get_mean_f0(path))






## AudioSegment.from_file

# import numpy as np
# from pydub import AudioSegment
 
audioseg = AudioSegment.from_file(path)
y = np.asarray(audioseg.get_array_of_samples())  # 将声音文件转换为数组格式
y = y if audioseg.channels == 1 else y.reshape(-1, 2)    # 若是双通道，则转换为（n,2）格式的数组
print('AudioSegment.from_file 声音数据的维度，数据类型，最大值，中间值，最小值为：', 
	y.shape, y.dtype, np.max(y), np.median(y), np.min(y))

## wavfile.read
# 有个博客说不推荐wave库方法
# https://blog.csdn.net/huplion/article/details/81040734
#不知道是不是指的是这个

# import numpy as np
# from scipy.io import wavfile
 
sr, y = wavfile.read(path)
print('wavfile.read 声音数据的维度，数据类型，最大值，中间值，最小值为：', 
	y.shape, y.dtype, np.max(y), np.median(y), np.min(y))


## sf.read

# import numpy as np
# import soundfile as sf
 
y, sr = sf.read(path)
print('sf.read 声音数据的维度，数据类型，最大值，中间值，最小值为：', 
	y.shape, y.dtype, np.max(y), np.median(y), np.min(y))


