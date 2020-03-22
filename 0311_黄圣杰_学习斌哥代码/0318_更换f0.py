import pyworld as pw
import librosa
import numpy as np
import soundfile as sf

oripath = 'E10001.wav'
#aimpath = 'E20018.wav'
aimpath = 'M10043.wav'

def get_sp(fpath:str):
    wav,sr = librosa.load(fpath,sr=None,mono=True,dtype=np.float64)
    f0,timeaxis = pw.harvesr(wav,sr)
    sp = pw.cheaptrick(wav,f0,timeaxis,sr,fft_size=1024)
    return sp  # 未code编码的sp


def get_mean_f0(fpath:str):  # 返回一个数字
    wav, SR = librosa.load(fpath, sr=None, mono=True, dtype=np.float64)  # librosa.load 返回音频信号值 & 采样率
    here_f0, timeaxis = pw.harvest(wav, SR)  # f0是一维数组，每帧会有一个f0
    print('line20:wav.shape = '+str(wav.shape)+'len(timeaxis) = '+str(len(timeaxis)))  # len(timeaxis)表示帧数
    print('here_f0.shape = '+str(here_f0.shape))  # =(1241,)

    sum_BigThan0_Logf0  = 0
    num_CorrectFrame = 0
    for i in range(len(timeaxis)):
        if here_f0[i]>0:
            tmp_log_f0 = np.log(here_f0[i])  # 别忘记【i】
            sum_BigThan0_Logf0 += tmp_log_f0
            num_CorrectFrame += 1

    mean_BigThan0_Logf0 = sum_BigThan0_Logf0 / num_CorrectFrame

    print('line33:mean_BigThan0_Logf0 = '+str(mean_BigThan0_Logf0)+'num_CorrectFrame = '+str(num_CorrectFrame))
    # line33:mean_BigThan0_Logf0 = 5.411331237332245 num_CorrectFrame = 993
    print('mean_BigThan0_Logf0.shape = '+str(mean_BigThan0_Logf0.shape))  # （）
    return mean_BigThan0_Logf0  # 对数形式的均值 ，一个数字       



    # return sum(here_f0) / len(timeaxis)


def main():
    _x, _fs = sf.read(oripath)  # 原始 音频信息&采样率
    _f0, _sp, _ap = pw.wav2world(_x, _fs)  # 原始f0,sp,ap,合成要用到，这个函数更 直接简单
    ori_f0, ori_timeaxis = pw.harvest(_x, _fs)  # 貌似只有这个函数能出来 timeaxis：对应帧信息



    x, fs = sf.read(oripath)  # 目标 音频信息&采样率
    f0, sp, ap = pw.wav2world(x, fs)  # 目标 f0,sp,ap
    aim_f0, aim_timeaxis = pw.harvest(x, fs)


    aim_mean_f0 = get_mean_f0(aimpath)  #目标f0均值,是 对数形式 的 数字(拿到下面去正态分布出来)
    ori_mean_f0 = get_mean_f0(oripath)

    #下面开始把源说话人的f0(有效帧：f0>0的帧)，做一个转换（先取对数，再加上两者 f0对数均值差 ）
    for i in range(len(ori_timeaxis)):  # 对原始说话人，逐帧筛选，有效帧 做对数处理后 再转换；
        if ori_f0[i] > 0:
            tmp_log_f0 = np.log(ori_f0[i])
            tmp_log_f0 = tmp_log_f0 - ori_mean_f0 + aim_mean_f0
            tmp_exp_f0 = np.exp(tmp_log_f0)  # 反对数
            ori_f0[i] = tmp_exp_f0


    # 这样说是不对，再来一版本：下面这行不行
    # aim_new_f0 = np.random.normal(aim_mean_f0, 1.0, sp.shape[0])  # 要的是目标新f0


    print('原始 _x：wav的尺寸_x.shape = '+str(_x.shape))  # 54852 维度，帧长度可以用len(timeaxis)

    print('目标：sp.shape[0] = '+str(sp.shape[0])+' sp.shape = '+str(sp.shape))
    print('目标 ap.shape = '+str(ap.shape))
    print('f0.shape = '+str(f0.shape)+'_f0.shape = '+str(_f0.shape))
    
    #print('aim_new_f0.shape'+str(aim_new_f0.shape))

    print('原始f0.shape = '+str(f0.shape))
    print('原始_sp.shape = '+str(_sp.shape)+' 原始_ap.shape = '+str(_ap.shape))


    synthesized = pw.synthesize(ori_f0, _sp, _ap, _fs, pw.default_frame_period)
    sf.write('./synthesized.wav', synthesized, _fs)


if __name__ == '__main__':
    main()




