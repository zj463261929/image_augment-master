#coding=utf-8
import codecs
import os
import numpy as np
import sys
import shutil  


image_dir = "/opt/ligang/data/icdar2017rctw/train-word_augment/" #要合并的文件夹，将文件夹下面所有文件合并,   改
save_dir = "/opt/ligang/data/icdar2017rctw/all/"                 #合并后保存的位置,  改
isRecognition = True   #True表拷贝的样本是识别，False为检测,  改

if os.path.exists(save_dir):
	shutil.rmtree(save_dir)  #删除save_dir指定的文件夹,目的是为了清空该文件夹

os.mkdir(save_dir) 
if not os.path.exists(image_dir) or not os.path.exists(save_dir):
	print ("Folder does not exist！(%s) or (%s)" % (image_dir, save_dir))
	sys.exit()

#
if isRecognition:  #识别
	txt_path = save_dir+"label.txt"
	if os.path.exists(save_dir + "img"):
		shutil.rmtree(save_dir + "img")
	os.mkdir(save_dir + "img")
		
	fw = codecs.open(txt_path, 'w', "utf-8")  #"a" 追加写
	for dirpath, dirnames, filenames in os.walk(image_dir):
		#print dirpath
		for filename in filenames:
			#print filename
			if filename.endswith('.txt'): #txt
				#print filename
				with codecs.open(image_dir+filename, 'rb', "utf-8") as ann_file:
					lines = ann_file.readlines()
					for l in lines:
						fw.write(l)
			if filename.endswith('.jpg'):
				shutil.copy(dirpath+"/"+filename, save_dir + "img/")
			
	fw.close()	
	
else: #检测
	if os.path.exists(save_dir + "img"):
		shutil.rmtree(save_dir + "img") #os.rmdir(save_dir + "img")
	os.mkdir(save_dir + "img")
	
	if os.path.exists(save_dir + "xml"):
		shutil.rmtree(save_dir + "xml")
	os.mkdir(save_dir + "xml")
	
	for dirpath, dirnames, filenames in os.walk(image_dir):
		for filename in filenames:
			if filename.endswith('.jpg'):
				shutil.copy(dirpath+"/"+filename, save_dir + "img/")
			if filename.endswith('.xml'):
				shutil.copy(dirpath+"/"+filename, save_dir + "xml/")
	
			
	