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

fun = FUN.Function()

#step1：设置路径、使用算法  
image_dir = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_7250/"  #原始图片路径
xml_dir = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/Annotations_7250/"  #图片对应xml的路径
image_txt = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets_7250/Main/train5.txt" #原始图片 标签对应的txt，比如文字识别：1.jpg 总结会，
																	#备注：如果处理的是文件夹下面的图片，没有这个txt文件，代码会自动生成的 

save_img_dir = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_7250_aug/"
save_xml_dir = "/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/Annotations_7250_aug/" #处理后保存的路径，每个增强方法会保存在对应文件夹下

#method_all = ["inverse", "elastic", "blur", "noise", "flip", "rotate", "translate", "zoom", "invertColor", "enhance", "contrastAndBright"]
method = ["inverse", "enhance", "contrastAndBright", "invertColor", "elastic", "blur", "noise", "flip", "rotate", "translate", "zoom"] 

#step2:创建文件夹
'''for i in range(len(method)):
	s = save_dir + method[i]
	ss = s + "/img"
	if not os.path.exists(ss):
		os.makedirs(ss)# 创建目录,这两个函数之间最大的区别是当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录。
	ss = s + "/xml"
	if not os.path.exists(ss):
		os.makedirs(ss)
	
#step3：处理
if not os.path.exists(image_dir) or not os.path.exists(save_dir) or not os.path.exists(xml_dir):
	print ("Folder does not exist！(%s) or (%s) or (%s)" % (image_dir, xml_dir, save_dir))
	sys.exit()'''

'''	
if not os.path.exists(image_txt):
	if image_dir.endswith("/"):
		image_txt = image_dir[:len(image_dir)-1] + ".txt"
		#print image_txt
	fw = codecs.open(image_txt, 'w', "utf-8")
	for dirpath, dirnames, filenames in os.walk(image_dir):
		for filename in filenames:
			if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):
				#img_path = image_dir + filename
				fw.write(filename + "\n")'''
print image_txt

num = 0
#处理txt对应的图片
with codecs.open(image_txt, 'rb', "utf-8") as ann_file:
	lines = ann_file.readlines()
	print (len(lines))
	for l in lines:
		lst = l.strip().split()
		#print lst[0]
		img_name11 = lst[0]+'.jpg'
		img_path = image_dir + img_name11
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
		print num
		xmin,ymin,xmax,ymax = read_xml( xml_path )
		'''img2 = img.copy()
		for i in range(len(xmin)):
			cv2.rectangle (img2,(xmin[i],ymin[i]),(xmax[i],ymax[i]),(0,0,255),1)
		cv2.imwrite("test.jpg", img2)'''
			
		w = img.shape[1]
		h = img.shape[0]
		

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
		if 0: #增强
			if 0 == random.randint(0, 2):
				result = fun.enhance(img, 0) #type=0(直方图均衡化) type=3(gamma变换)
			else:
				result = fun.enhance(img, 3)
				
			if result is None:
				continue
			
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_enhance.jpg"
			save_path = save_dir + "enhance/img/" + img_name1
			#print "\n"
			#print save_path
			save_path_xml = save_dir + "enhance/xml/" + img_name[:len(img_name)-4] + "_enhance.xml"
			cv2.imwrite(save_path, result)
			write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
		a=np.random.rand(1)
		b=0
		if a>0.8:
			b=1
		else:
			b=0			
		if b: #对比度、亮度
			mean = getImgMean(img)
			alpha = (1.8-0.0)*(1-mean/255)
			beta = random.randint(-10, -1)
			
			result = fun.contrastAndBright(img, alpha, beta)
			if result is None:
				continue
			
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_contrastAndBright.jpg"
			save_path = save_img_dir + img_name1
			save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_contrastAndBright.xml"
			cv2.imwrite(save_path, result)
			write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
			
		if 0: #颜色变换，针对彩色，只改变hsv的h分量
			min_h = -5
			max_h = 5
			result = fun.invertColor(img, min_h, max_h)
			if result is None:
				continue
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_invertColor.jpg"
			save_path = save_dir + "invertColor/img/" + img_name1
			save_path_xml = save_dir + "invertColor/xml/" + img_name[:len(img_name)-4] + "_invertColor.xml"
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
		if b: #模糊
			type = random.randint(0, 2)
			result = fun.blur(img, kernel_dim=5, sigma=1.5, type=type) #type:（0=blur,1= GaussianBlur, 2= medianBlur）
			if result is None:
				continue
			
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_blur.jpg"
			save_path = save_img_dir + img_name1
			save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_blur.xml"
			cv2.imwrite(save_path, result)
			write_xml(xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
			
		a=np.random.rand(1)
		b=0
		if a>0.8:
			b=1
		else:
			b=0			
		if b: #噪音
			type = random.randint(0, 2)
			result = fun.noise(img, param=0.3, type=type) #type:（0=高斯,1=椒盐噪声）
			if result is None:
				continue
			
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_noise.jpg"
			save_path = save_img_dir + img_name1
			save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_noise.xml"
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
		if b: #旋转,改
			angle = random.randint(-5,5)#random.random()*30 #0~5, 角度
			type=1 #改
			scale = 1.0
			result = fun.rotate(img, angle, scale=scale, type=type) #type:（0=旋转不带放大,1=旋转带放大）
			if result is None:
				continue
			
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_rotate.jpg"
			save_path = save_img_dir + img_name1
			save_path_xml = save_xml_dir + img_name[:len(img_name)-4] + "_rotate.xml"
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
		
		if 0: #平移
			x_trans = random.randint(0,10)
			y_trans = random.randint(0,10)
			#print x_trans,y_trans
			type = 1
			result = fun.translate(img, x_trans, y_trans, type=type) #type:（0=平移后大小不变,1=平移后大小改变,从图的下方、右边裁剪）
			if result is None:
				continue
				
			image_height = result.shape[0]
			image_width = result.shape[1]
			
			img_name1 = img_name[:len(img_name)-4] + "_translate.jpg"
			save_path = save_dir + "translate/img/" + img_name1
			save_path_xml = save_dir + "translate/xml/" + img_name[:len(img_name)-4] + "_translate.xml"
			cv2.imwrite(save_path, result)
			
			if 0==type:
				write_xml( xml_path, save_path_xml, xmin,ymin,xmax,ymax, image_width, image_height)
			if 1==type:
				xmin1,ymin1,xmax1,ymax1 = get_translate(img, x_trans, y_trans, xmin,ymin, xmax, ymax)
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
				
				
#img = cv2.imread("6.jpg")#("1.jpg", -1)
#cv2.imwrite("1gray.jpg", img)

#result = fun.inverse(img)
#result = fun.elastic(img, 31, 12, 47) #
#result = fun.blur(img,9,1.5,0)
#result = fun.flip(img,-1)
#result = fun.noise(img,0.5,1)
#result = fun.rotate(img, 30, 1.0, 1)
#result = fun.translate(img,20,100,1)
#result = fun.zoom(img,200,0,300,300)
#result = fun.invertColor(img, -20, 20, 1)
#result = fun.enhance(img, 3) 
#result = fun.contrastAndBright(img, 1, 50)
#cv2.imwrite("2_5_.jpg", result)
