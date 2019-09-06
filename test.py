import os 
import unicodedata as u
import io
import re

os.path.join('/home/siplab/Desktop/mannkibaat/telugu/text')
file_name = '/home/siplab/Desktop/mannkibaat/telugu/text/telugu_jun_18.txt'
f = io.open(file_name,'r',encoding = 'utf8')

for line in f.readlines():
	k = re.split('; |,|\\.|!|\?|\s|\n',line)
	l = len(k)
	for a in k:
		for i in a:
			if ((ord(i)<3072)or(ord(i)>3199)):
				print(a)
				break
f.close()
