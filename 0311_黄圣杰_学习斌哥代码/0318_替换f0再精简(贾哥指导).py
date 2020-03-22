import pyworld as pw
import librosa
import numpy as np
import soundfile as sf
oripath = 'E10001.wav'
aimpath = 'M10043.wav'

def wavToWorldFeature(fpath:str):
    wav,SR = librosa.load(fpath,sr = None,mono = True,dtype = np.float64)
    return(pw.wav2world(wav, SR))


def get_mean_f0(fpath:str):  # 返回一个数字 -> return mean_f0,wav,SR,here_f0,sp,ap
	wav,SR = librosa.load(fpath,sr = None,mono = True,dtype = np.float64)
	here_f0, sp, ap = pw.wav2world(wav, SR)
	mean_f0 = np.mean(np.log(here_f0[ here_f0 > 0 ]))  # Good,学会这种“串式”写法，多学几次就会了，line16也是
	return mean_f0,wav,SR,here_f0,sp,ap

def main():
    aim_mean_f0,_x,_fs,ori_f0,_sp,_ap = get_mean_f0(aimpath)  #目标f0均值,是 对数形式 的 数字(拿到下面去正态分布出来)
    ori_mean_f0,x,fs,f0,sp,ap = get_mean_f0(oripath)   # return mean_f0,wav,SR,here_f0,sp,ap
    

    ori_f0[ ori_f0 > 0 ] = np.exp(np.log(ori_f0[ ori_f0 > 0 ]) - ori_mean_f0 + aim_mean_f0)	 # 下面开始把源说话人的f0(有效帧：f0>0的帧)，做一个转换（先取对数，再加上两者 f0对数均值差 ）
    synthesized = pw.synthesize(ori_f0, _sp, _ap, _fs, pw.default_frame_period)
    sf.write('./synthesized.wav', synthesized, _fs)

if __name__ == '__main__':
    main()
    print(type(wavToWorldFeature(oripath)))
    print(len(wavToWorldFeature(oripath)))






