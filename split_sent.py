import io
import re
import os
from scipy.io import wavfile as w
os.mkdir('telugu_december_2018')

f = io.open('telugu_december_2018_sentenc1.txt','r',encoding = 'utf-8')
fs,d = w.read('telugu_december_2018.wav')
i=0
for line in f.readlines():
	spl = line.split()
	strt_time = float(spl[0])
	end_time = float(spl[1])
	strt_samp = int(strt_time*fs)
	end_samp = int(end_time*fs)
	fname  = 'telugu_december_2018_'+i
	print(fname)	