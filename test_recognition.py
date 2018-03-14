#coding=utf-8
import codecs
import random
import cv2, os
import sys #sys是Python内建标准库
sys.path.append(os.getcwd())
from function import *
import function as FUN

fun = FUN.Function()

#step1：设置路径、使用算法  
image_dir = "/opt/zhangjing/ocr/augmented_dataset/image_augment-master/img_recognition/"  #原始图片路径
image_txt = "/opt/zhangjing/ocr/augmented_dataset/image_augment-master/img_recognition.txt" #原始图片 标签对应的txt，比如文字识别：1.jpg 总结会，
																	#备注：如果处理的是文件夹下面的图片，没有这个txt文件，代码会自动生成的 
save_dir = "/opt/zhangjing/ocr/augmented_dataset/image_augment-master/result_recognition/" #处理后保存的路径，每个增强方法会保存在对应文件夹下

#method_all = ["inverse", "elastic", "blur", "noise", "flip", "rotate", "translate", "zoom", "invertColor", "enhance", "contrastAndBright"]
method = ["inverse", "enhance", "contrastAndBright", "invertColor", "elastic", "blur", "noise", "flip", "rotate", "translate", "zoom"] 

#step2:创建文件夹、写文件
fw_inverse = codecs.open(save_dir + "inverse.txt", 'w', "utf-8")  #"a" 追加写
fw_elastic = codecs.open(save_dir + "elastic.txt", 'w', "utf-8")
fw_blur = codecs.open(save_dir + "blur.txt", 'w', "utf-8")
fw_noise = codecs.open(save_dir + "noise.txt", 'w', "utf-8")
fw_flip = codecs.open(save_dir + "flip.txt", 'w', "utf-8")
fw_rotate = codecs.open(save_dir + "rotate.txt", 'w', "utf-8")
fw_translate = codecs.open(save_dir + "translate.txt", 'w', "utf-8")
fw_zoom = codecs.open(save_dir + "zoom.txt", 'w', "utf-8")
fw_invertColor = codecs.open(save_dir + "invertColor.txt", 'w', "utf-8")
fw_enhance = codecs.open(save_dir + "enhance.txt", 'w', "utf-8")
fw_contrastAndBright = codecs.open(save_dir + "contrastAndBright.txt", 'w', "utf-8")

for i in range(len(method)):
	s = save_dir + method[i]
	if not os.path.exists(s):
		os.mkdir(s)# 创建目录,这两个函数之间最大的区别是当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录。

	
#step3：处理
if not os.path.exists(image_dir) or not os.path.exists(save_dir):
	print ("Folder does not exist！(%s) or (%s)" % (image_dir, save_dir))
	sys.exit()
	
if not os.path.exists(image_txt):
	if image_dir.endswith("/"):
		image_txt = image_dir[:len(image_dir)-1] + ".txt"
		#print image_txt
	fw = codecs.open(image_txt, 'w', "utf-8")
	for dirpath, dirnames, filenames in os.walk(image_dir):
		for filename in filenames:
			if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):
				#img_path = image_dir + filename
				fw.write(filename + "\n")

#处理txt对应的图片
with codecs.open(image_txt, 'rb', "utf-8") as ann_file:
	lines = ann_file.readlines()
	for l in lines:
		lst = l.strip().split()
		#print lst[0]
		if len(lst)>0:
			img_name = lst[0]
			img_path = image_dir + img_name
			print img_path
			img = cv2.imread(img_path)

			if img is None:
				continue
				
			if 1: #表示处理, 反色
				result = fun.inverse(img)
				if result is None:
					continue

				img_name1 = img_name[:len(img_name)-4] + "_inverse.jpg"
				save_path = save_dir + "inverse/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				#print lst1
				fw_inverse.write(" ".join(lst1) + "\n")
				
			if 1: #增强
				if 0 == random.randint(0, 1):
					result = fun.enhance(img, 0) #type=0(直方图均衡化) type=3(gamma变换)
				else:
					result = fun.enhance(img, 3)
					
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_enhance.jpg"
				save_path = save_dir + "enhance/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_enhance.write(" ".join(lst1) + "\n")
				
			if 1: #对比度、亮度
				mean = getImgMean(img)
				alpha = (3.0-0.0)*(1-mean/255)
				beta = random.randint(-20, 20)
				
				result = fun.contrastAndBright(img, alpha, beta)
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_contrastAndBright.jpg"
				save_path = save_dir + "contrastAndBright/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_contrastAndBright.write(" ".join(lst1) + "\n")
				
			if 1: #颜色变换，针对彩色，只改变hsv的h分量
				min_h = -5
				max_h = 5
				result = fun.invertColor(img, min_h, max_h)
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_invertColor.jpg"
				save_path = save_dir + "invertColor/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_invertColor.write(" ".join(lst1) + "\n")   
			
			if 1: #弹性变形，主要针对手写字体
				result = fun.elastic(img, kernel_dim=31, sigma=6, alpha=47)
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_elastic.jpg"
				save_path = save_dir + "elastic/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_elastic.write(" ".join(lst1) + "\n")   

			if 1: #模糊
				type = random.randint(0, 2)
				result = fun.blur(img, kernel_dim=5, sigma=1.5, type=type) #type:（0=blur,1= GaussianBlur, 2= medianBlur）
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_blur.jpg"
				save_path = save_dir + "blur/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_blur.write(" ".join(lst1) + "\n")   
				
			if 1: #噪音
				type = random.randint(0, 1)
				result = fun.noise(img, param=0.1, type=type) #type:（0=高斯,1=椒盐噪声）
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_noise.jpg"
				save_path = save_dir + "noise/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_noise.write(" ".join(lst1) + "\n")
				
			if 1: #镜像
				result = fun.flip(img, flipCode=0) #flipCode:（0=沿x轴翻转(上下),1=沿y轴翻转(左右),2=180旋转）
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_flip.jpg"
				save_path = save_dir + "flip/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_flip.write(" ".join(lst1) + "\n")
				
			if 1: #旋转
				angle = random.randint(-5,5) #0~5,角度
				result = fun.rotate(img, angle, scale=1.0, type=0) #type:（0=旋转不带放大,1=旋转带放大）
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_rotate.jpg"
				save_path = save_dir + "rotate/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_rotate.write(" ".join(lst1) + "\n")
			
			if 1: #平移
				x_trans = random.randint(0,10)
				y_trans = random.randint(0,10)
				result = fun.translate(img, x_trans, y_trans, type=1) #type:（0=平移后大小不变,1=平移后大小改变）
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_translate.jpg"
				save_path = save_dir + "translate/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_translate.write(" ".join(lst1) + "\n")
				
			if 1: #变焦/拉伸
				p1x = random.randint(0,10)
				p1y = random.randint(0,10)
				p2x = img.shape[1] -1 - p1x
				p2y = img.shape[0] -1 - p1y
				result = fun.zoom(img, p1x, p1y, p2x, p2y)
				if result is None:
					continue
				img_name1 = img_name[:len(img_name)-4] + "_zoom.jpg"
				save_path = save_dir + "zoom/" + img_name1
				cv2.imwrite(save_path, result)
				lst1 = []
				lst1.append(img_name1)
				lst1[1:] = lst[1:]
				fw_zoom.write(" ".join(lst1) + "\n")
				
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
