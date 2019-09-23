import os 
from librosa.feature import melspectrogram
from scipy.io import wavfile as w

filelist = os.listdir(os.curdir)
dirlist = [i for i in filelist if os.path.isdir(os.path.join(os.curdir,i))]
for dirname in dirlist:
	l = os.listdir(dirname)
	i = 1
	for filename in l:
		if(filename.endswith('.wav')):
			fs,d = w.read(os.path.join(dirname,filename))
			d = np.array(d,dtype = float)
			d /= 32,767
			m = melspectrogram(d,fs,n_mels = 80,n_fft = 400,hop_length = 160,fmax = 8000)
			fspec = io.open(dirname+'spec','w')
			for row in m:
				for j in row:
					fspec.write(str(j))
				fspec.write('\n')
			i = i+1
