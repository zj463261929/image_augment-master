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
import threading

def get_index():
	a=np.random.rand(1)
	b=0
	if a>0.6:
		b=1
	else:
		b=0
	return b


g1=0 #增强
g2=0 #对比度
g3=0 #invertColor
g4=0 #模糊
g5=0 #噪音
g6=0 #旋转
g7=1 #平移

loopNum=1

def img_aug(image_dir,xml_dir,image_txt,save_img_dir,save_xml_dir):


	fun = FUN.Function()	

	#method_all = ["inverse", "elastic", "blur", "noise", "flip", "rotate", "translate", "zoom", "invertColor", "enhance", "contrastAndBright"]
	method = ["inverse", "enhance", "contrastAndBright", "invertColor", "elastic", "blur", "noise", "flip", "rotate", "translate", "zoom"] 


	print image_txt
	#print 'thread %s starts.' % threading.current_thread().name

	num = 0
	#处理txt对应的图片
	with codecs.open(image_txt, 'rb', "utf-8") as ann_file:
		lines = ann_file.readlines()
		print (len(lines))
		for l in lines:
			lst = l.strip().split()
			print lst[0]
			img_name11 = lst[0]+'.jpg'
			img_path = image_dir + img_name11
			if not os.path.exists(img_path):
				continue
				
			#print img_path
			img = cv2.imread(img_path)

			lst1 = img_name11.strip().split("/")
			
			img_name = lst1[0]
			
			#index = img_name[:len(img_name)-4]
			xml_path = xml_dir + lst[0]+'.xml' #index + ".xml"
			#print xml_path
			if not os.path.exists(xml_path):
				continue
			
			if img is None:
				continue
				
			num = num + 1
			#print num
			xmin,ymin,xmax,ymax = read_xml( xml_path )
			'''img2 = img.copy()
			for i in range(len(xmin)):
				cv2.rectangle (img2,(xmin[i],ymin[i]),(xmax[i],ymax[i]),(0,0,255),1)
			cv2.imwrite("test.jpg", img2)'''
				
			w = img.shape[1]
			h = img.shape[0]
			
			result = None
			#随机选择增强方式进行增强
			if 0: #表示处理, 反色
				result = fun.inverse(img)
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_inverse.jpg"
				save_path = save_dir + "inverse/img/" + img_name1
				save_path_xml = save_dir + "inverse/xml/" + img_name[:len(img_name)-4] + "_inverse.xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
				
			a=np.random.rand(1)
			b=0
			if a>0.8:
				b=1
			else:
				b=0
			if g1 and b: #增强
				f=get_index()
				if f and (result is not None):
					img = result.copy()

				if 0:#0 == random.randint(0, 2):
					result = fun.enhance(img, 0) #type=0(直方图均衡化) type=3(gamma变换)
				else:
					result = fun.enhance(img, 3)
					
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_enhance" + str(loopNum) + ".jpg"
				save_path = save_img_dir + img_name1
				#print "\n"
				#print save_path
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_enhance" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
			
			a=np.random.rand(1)
			b=0
			if a>0.8:
				b=1
			else:
				b=0			
			if g2 and b: #b: #对比度、亮度
				mean = getImgMean(img)
				c=random.randint(0, 2)
				
				if c==0 and 1:
					alpha = (1.8-0.0)*(1-mean/255)
					beta = random.randint(-10, -1)
				elif c==1 and 0:
					alpha = (3.0-0.0)*(1-mean/255)
					beta = random.randint(-20, 20)
				elif c==2 and 0:
					alpha = (3.0-0.5)*(1-mean/255)
					beta = random.randint(-10, 10)
				
				f=get_index()
				if f and (result is not None):
					img = result.copy()
					
				result = fun.contrastAndBright(img, alpha, beta)
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_contrastAndBright" + str(loopNum) + ".jpg"
				save_path = save_img_dir + img_name1
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_contrastAndBright" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
				
			if g3 and get_index(): #颜色变换，针对彩色，只改变hsv的h分量
				f=get_index()
				if f and (result is not None):
					img = result.copy()
					
				min_h = -5
				max_h = 5
				result = fun.invertColor(img, min_h, max_h)
				if result is None:
					continue
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_invertColor" + str(loopNum) + ".jpg"
				save_path = save_img_dir +  img_name1
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_invertColor" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
			
			if 0: #弹性变形，主要针对手写字体
				result = fun.elastic(img, kernel_dim=31, sigma=6, alpha=47)
				if result is None:
					continue
					
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_elastic.jpg"
				save_path = save_dir + "elastic/img/" + img_name1
				save_path_xml = save_dir + "elastic/xml/" + img_name[:len(img_name)-4] + "_elastic.xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
				
			a=np.random.rand(1)
			b=0
			if a>0.8:
				b=1
			else:
				b=0
			if g4 and b: #模糊
				f=get_index()
				if f and (result is not None):
					img = result.copy()
					
				type = random.randint(0, 2)
				result = fun.blur(img, kernel_dim=5, sigma=1.5, type=type) #type:（0=blur,1= GaussianBlur, 2= medianBlur）
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_blur" + str(loopNum) + ".jpg"
				save_path = save_img_dir + img_name1
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_blur" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
				
			a=np.random.rand(1)
			b=0
			if a>0.8:
				b=1
			else:
				b=0			
			if g5 and b: #噪音
				f=get_index()
				if f and (result is not None):
					img = result.copy()
					
				type = 0#random.randint(0, 2)
				
				result = fun.noise(img, param=0.03, type=type) #type:（0=高斯,1=椒盐噪声）
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_noise" + str(loopNum) + ".jpg"
				save_path = save_img_dir + img_name1
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_noise" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
				
			if 0: #镜像
				flipCode = 1
				result = fun.flip(img, flipCode=flipCode) #flipCode:（0=沿x轴翻转(上下),>0 沿y轴翻转(左右),<0 180旋转）
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_flip.jpg"
				save_path = save_dir + "flip/img/" + img_name1
				save_path_xml = save_dir + "flip/xml/" + img_name[:len(img_name)-4] + "_flip.xml"
				cv2.imwrite(save_path, result)
			
				xmin1,ymin1,xmax1,ymax1 = get_flip(img, flipCode, xmin,ymin, xmax, ymax)
				write_xml( xml_path, save_path_xml, xmin1,ymin1,xmax1,ymax1, image_width, image_height)
				
			a=np.random.rand(1)
			b=0
			if a>0.8:
				b=1
			else:
				b=0				
			if g6 and b: #旋转,改
				f=get_index()
				if f and (result is not None):
					img = result.copy()
					
				angle = random.randint(-2,2)#random.randint(-5,5)#random.random()*30 #0~5, 角度
				type=1 #改
				scale = 1.0
				result = fun.rotate(img, angle, scale=scale, type=type) #type:（0=旋转不带放大,1=旋转带放大）
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_rotate" + str(loopNum) + ".jpg"
				save_path = save_img_dir + img_name1
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_rotate" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				
				#为了查看旋转后的效果，将XML的结果画在图中验证
				'''result1=img.copy()
				for i in xrange(len(xmin)):
					cv2.rectangle(result1, (xmin[i],ymin[i]), (xmax[i],ymax[i]), (255, 255, 255), 2)
				cv2.imwrite('org.jpg', result1)'''
				xmin1,ymin1,xmax1,ymax1 = get_rotate_change(img,result,angle, xmin, ymin, xmax,ymax,scale)
				write_xml( xml_path, save_path_xml, xmin1,ymin1,xmax1,ymax1, image_width, image_height)
				#为了查看旋转后的效果，将XML的结果画在图中验证
				'''result2=result.copy()
				for i in xrange(len(xmin)):
					cv2.rectangle(result2, (xmin1[i],ymin1[i]), (xmax1[i],ymax1[i]), (255, 255, 255), 2)
				cv2.imwrite('rota1.jpg', result2)'''
			
			if g7 and get_index(): #平移
				f=get_index()
				if f and (result is not None):
					img = result.copy()
					
				x_trans = random.randint(0,10)
				y_trans = random.randint(0,10)
				#print x_trans,y_trans
				type = 0
				result = fun.translate(img, x_trans, y_trans, type=type) #type:（0=平移后大小不变,1=平移后大小改变,从图的下方、右边裁剪）
				if result is None:
					continue
					
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_translate" + str(loopNum) + ".jpg"
				save_path = save_img_dir + img_name1
				save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_translate" + str(loopNum) + ".xml"
				cv2.imwrite(save_path, result)
				#print (x_trans,y_trans)
				xmin1,ymin1,xmax1,ymax1 = get_translate(img, x_trans, y_trans, xmin,ymin, xmax, ymax,type)
				if 0==type:
					write_xml( xml_path, save_path_xml, xmin1,ymin1,xmax1,ymax1, image_width, image_height)
				if 1==type:
					write_xml( xml_path, save_path_xml, xmin1,ymin1,xmax1,ymax1, image_width, image_height)
				
			if 0: #变焦/拉伸
				p1x = random.randint(0,10)
				p1y = random.randint(0,10)
				#print p1x,p1y
				p2x = img.shape[1] - 1 - p1x
				p2y = img.shape[0] - 1 - p1y
				result = fun.zoom(img, p1x, p1y, p2x, p2y)#按照区域(p1x, p1y, p2x, p2y)crop，并放大到原图
				if result is None:
					continue
				
				image_height = result.shape[0]
				image_width = result.shape[1]
				
				img_name1 = img_name[:len(img_name)-4] + "_zoom.jpg"
				save_path = save_dir + "zoom/img/" + img_name1
				save_path_xml = save_dir + "zoom/xml/" + img_name[:len(img_name)-4] + "_zoom.xml"
				cv2.imwrite(save_path, result)
			
				xmin1,ymin1,xmax1,ymax1 = get_zoom(img, result, p1x, p1y, p2x, p2y, xmin,ymin, xmax, ymax)
				write_xml( xml_path, save_path_xml, xmin1,ymin1,xmax1,ymax1, image_width, image_height)

				