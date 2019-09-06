# -*- coding: utf-8 -*-
import os 
import unicodedata as u
import io
import re

os.path.join('/home/siplab/Desktop/mannkibaat/english/text')
file_name = '/home/siplab/Desktop/mannkibaat/telugu/text/telugu_aug_18.txt'
f = io.open(file_name,'r',encoding = 'utf-8')
fw = io.open('text.txt','w',encoding = 'utf-8')

for i in f.readlines():
	b = re.sub("\\.+\s*|\?|\!|\-|\'|\–|,|\/|:"," ",i)
	b = b.replace(u'\u201c', ' ').replace(u'\u201d', ' ')
	b = b.replace(u'\u2019', ' ').replace(u'\u2018', ' ')	
	b = b.replace(u'\u2013', ' ')
	fw.write(b)
	
	#print(b)

