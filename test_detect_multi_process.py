#coding=utf-8
import codecs
import random
import cv2, os
import sys #sys是Python内建标准库
sys.path.append(os.getcwd())
from function import *
import function as FUN
import math
import numpy as np
import time
from test_detect_single import img_aug
from multiprocessing import Process

image_dir = "/opt/zhangjing/Detectron/test/aug/"  #原始图片路径
xml_dir = "/opt/zhangjing/Detectron/test/aug/"  #图片对应xml的路径
image_txt1 = "/opt/zhangjing/Detectron/test/other.txt"#原始图片 标签对应的txt
'''image_txt2 = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/forklift.txt"#原始图片 标签对应的txt
image_txt3 = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/digger.txt"#原始图片 标签对应的txt
image_txt4 = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets_8558/Main/train_4.txt"#原始图片 标签对应的txt
image_txt5 = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets_8558/Main/train_5.txt"#原始图片 标签对应的txt
image_txt6 = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets_8558/Main/train_6.txt"#原始图片 标签对应的txt
'''
save_img_dir = "/opt/zhangjing/Detectron/test/aug_out/"
save_xml_dir = "/opt/zhangjing/Detectron/test/aug_out/" #处理后保存的路径，每个增强方法会保存在对应文件夹下
if not os.path.exists(save_xml_dir):
	os.mkdir(save_xml_dir)
if not os.path.exists(save_img_dir):
	os.mkdir(save_img_dir)

if __name__ == "__main__":
	print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	print 'Parent process %s starts.' % os.getpid()
	#创建线程
	my_process1 = Process(target = img_aug, args=(image_dir,xml_dir,image_txt1,save_img_dir,save_xml_dir) , name= 'ys_process1')
	
	'''my_process2 = Process(target = img_aug, args=(image_dir,xml_dir,image_txt2,save_img_dir,save_xml_dir) , name= 'ys_process2')
	
	my_process3 = Process(target = img_aug, args=(image_dir,xml_dir,image_txt3,save_img_dir,save_xml_dir) , name= 'ys_process3')
	
	my_process4 = Process(target = img_aug, args=(image_dir,xml_dir,image_txt4,save_img_dir,save_xml_dir) , name= 'ys_process4')
	
	my_process5 = Process(target = img_aug, args=(image_dir,xml_dir,image_txt5,save_img_dir,save_xml_dir) , name= 'ys_process5')
	
	my_process6 = Process(target = img_aug, args=(image_dir,xml_dir,image_txt6,save_img_dir,save_xml_dir) , name= 'ys_process6')
	'''
	#等待2s
	time.sleep(2)
	#启动线程
	my_process1.start()
	'''my_process2.start()
	my_process3.start()
	my_process4.start()
	my_process5.start()
	my_process6.start()'''
	#等待线程结束
	my_process1.join()
	'''my_process2.join()
	my_process3.join()
	my_process4.join()
	my_process5.join()
	my_process6.join()'''
	print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	print 'process %s ends.' % os.getpid()

