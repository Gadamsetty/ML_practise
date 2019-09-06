# -*- coding: utf-8 -*-
import os 
import unicodedata as u
import io
import re
import num2words
p = '/home/siplab/Desktop/mannkibaat/english/text_w'
p_w = '/home/siplab/Desktop/mannkibaat/english/comp'
for filename in os.listdir(p):
	f = io.open(os.path.join(p,filename),'r+',encoding='utf-8')
	fw = io.open(os.path.join(p_w,filename),'w',encoding='utf-8')
	for line in f.readlines():
		line  = re.sub(r"(\d+)",lambda x: num2words.num2words(int(x.group(0))),line)
		fw.write(line)
