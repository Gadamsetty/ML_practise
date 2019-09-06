# -*- coding: utf-8 -*-
import os 
import unicodedata as u
import io
import re

p = '/home/siplab/Desktop/mannkibaat/english/comp'
p_w = '/home/siplab/Desktop/mannkibaat/english/completed'
for filename in os.listdir(p):
	f = io.open(os.path.join(p,filename),'r+',encoding='utf-8')
	fw = io.open(os.path.join(p_w,filename),'w',encoding='utf-8')
	for i in f.readlines():
		b = re.sub("\\.+\s*|\?|\!|\-|\'|\–|,|\/|:|;|\(|\)|\|\""," ",i)
		b = b.replace(u'\u201c', ' ').replace(u'\u201d', ' ')
		b = b.replace(u'\u2019', ' ').replace(u'\u2018', ' ')	
		b = b.replace(u'\u2013', ' ')
		b = b.replace(u"\u0964",' ')
		b = b.replace(u"\u0022",' ')
		b = b.replace('&','and')
		fw.write(b)
