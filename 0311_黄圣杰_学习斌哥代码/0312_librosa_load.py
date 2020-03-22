import librosa
import numpy as np
import pyworld as pw
from timeit import default_timer as timer

file = "E10001.wav"
audio,sr = librosa.load(file,sr=24000,mono=True,dtype=np.float64)

# 参数		作用
# path		音频路径
# sr		采样率（默认22050，但是有重采样的功能）
# mono		设置为true是单通道，否则是双通道
# offset	音频读取的时间
# duration	获取音频的时长

# 函数返回值
# audio = y  : 音频的信号值，类型是 ndarray
# sr         : 采样率

start = timer()

f0,t = pw.harvest(audio,sr)  
# 这是一个用pyworld得到f0特征的方法；
# 输入：音频信号 & 采样率；输出：f0特征和对应的时间轴信息
sp = pw.cheaptrick(audio, f0, t, sr)  # extract smoothed spectrogram
ap = pw.d4c(audio, f0, t, sr)         # extract aperiodicity
end = timer()
print('Feature Extraction:', end - start, 'seconds')



print('\nsr = ' ) # 采样率sampling rate
print(sr)

print('\naudio = ' ) # 得到的音频信号值
print(audio)

print('\ntype_of_audio = ') # 信号类型：numpy.ndarray
print(type(audio))

print('\nlen_of_aodio = ' )
print(len(audio))

print('\nshape_of_audio = ' )
print(audio.shape)

