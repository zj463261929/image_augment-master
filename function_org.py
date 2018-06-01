#coding=utf-8
import re
from numpy import *
import numpy as np
import math
import cv2
from scipy.signal import convolve2d	 
import xml.dom.minidom

# See http://arxiv.org/pdf/1003.0358.pdf for the description of the method
def elastic_distortion(image, kernel_dim=31, sigma=6, alpha=47):

	# Returns gaussian kernel in two dimensions
	# d is the square kernel edge size, it must be an odd number.
	# i.e. kernel is of the size (d,d)
	def gaussian_kernel(d, sigma):
		if d % 2 == 0:
			raise ValueError("Kernel edge size must be an odd number")

		cols_identifier = np.int32(np.ones((d, d)) * np.array(np.arange(d)))
		rows_identifier = np.int32(np.ones((d, d)) * np.array(np.arange(d)).reshape(d, 1))

		kernel = np.exp(-1. * ((rows_identifier - d/2)**2 +
			(cols_identifier - d/2)**2) / (2. * sigma**2))
		kernel *= 1. / (2. * math.pi * sigma**2)  # normalize
		return kernel

	field_x = np.random.uniform(low=-1, high=1, size=image.shape) * alpha #原图大小
	field_y = np.random.uniform(low=-1, high=1, size=image.shape) * alpha
	#print field_x.shape
	kernel = gaussian_kernel(kernel_dim, sigma)
	#print kernel.shape
	
	# Distortion fields convolved with the gaussian kernel
	# This smoothes the field out.
	field_x = convolve2d(field_x, kernel, mode="same")
	field_y = convolve2d(field_y, kernel, mode="same")

	d = image.shape[0]
	cols_identifier = np.int32(np.ones((d, d))*np.array(np.arange(d)))
	rows_identifier = np.int32(np.ones((d, d))*np.array(np.arange(d)).reshape(d, 1))
	#print cols_identifier.shape #d*d

	down_row = np.int32(np.floor(field_x)) + rows_identifier #d*d
	top_row = np.int32(np.ceil(field_x)) + rows_identifier
	down_col = np.int32(np.floor(field_y)) + cols_identifier
	top_col = np.int32(np.ceil(field_y)) + cols_identifier
#	 plt.imshow(field_x, cmap=pylab.cm.gray, interpolation="none")
#	 plt.show()
	#print down_row.shape #原图大小

	padded_image = np.pad(image, pad_width=d, mode="constant", constant_values=0)#3d*3d, 填充，填充数值为0
	#print padded_image.shape
	#cv2.imwrite("22.jpg", padded_image)
	
	x1 = down_row.flatten() #(d*d)*1
	y1 = down_col.flatten()
	x2 = top_row.flatten()
	y2 = top_col.flatten()
	#print x1.shape
 
	Q11 = padded_image[d+x1, d+y1] #(d*d)*1
	Q12 = padded_image[d+x1, d+y2]
	Q21 = padded_image[d+x2, d+y1]
	Q22 = padded_image[d+x2, d+y2]
	#print Q11.shape
	x = (rows_identifier + field_x).flatten() #(d*d)*1
	y = (cols_identifier + field_y).flatten()
	#print x.shape

	# Bilinear interpolation algorithm is as described here:
	# https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
	distorted_image = (1. / ((x2 - x1) * (y2 - y1)))*(
		Q11 * (x2 - x) * (y2 - y) +
		Q21 * (x - x1) * (y2 - y) +
		Q12 * (x2 - x) * (y - y1) +
		Q22 * (x - x1) * (y - y1))

	distorted_image = distorted_image.reshape((d, d))
	return distorted_image
	
def getImgMean(img):
	mean =  cv2.mean(img)
	length = len(img.shape)
	sum = 0.0
	val = 128
	if 3==length:
		sum = sum + mean[0] + mean[1] + mean[2]
		val = sum/3
	else:
		val = mean[0]
	return val
			

class Function:
	def contrastAndBright(self,img, alpha, beta): #alpha=0.0~3.0, 
		height = img.shape[0]
		width = img.shape[1]
		img2 = np.zeros(img.shape)
		length = len(img.shape)
		
		if 3==length:
			for i in range(height):
				for j in range(width):
					img2[i,j] = (img[i,j][0]*alpha+beta,img[i,j][1]*alpha+beta,img[i,j][2]*alpha+beta) 
		else:
			for i in range(height):
				for j in range(width):
					img2[i,j] = (img[i,j]*alpha+beta) 
		return img2	
	
	def enhance(self, img, type=3):
		def equalizeHist(img):
			if 3==len(img.shape):
				b, g, r = cv2.split(img)  
				b1 = cv2.equalizeHist(b)
				g1 = cv2.equalizeHist(g)
				r1 = cv2.equalizeHist(r)
				return cv2.merge([b1,g1,r1])
			else:
				return cv2.equalizeHist(img)
				
		def laplace(img):
			kernel = np.array([[0,-1,0],[0,5,0],[0,-1,0]])
			return cv2.filter2D(img,-1,kernel)
			
		def log(img):
			height = img.shape[0]
			width = img.shape[1]
			length = len(img.shape)
			img2 = img.copy()
			if 3==length:
				for i in range(height):
					for j in range(width):
						img2[i,j] = (math.log(1+img[i,j][0]), math.log(1+img[i,j][1]), math.log(1+img[i,j][2]))
				cv2.normalize(img2,img2,0,255,cv2.NORM_MINMAX)
				return cv2.convertScaleAbs(img2)	
			else:
				for i in range(height):
					for j in range(width):
						img2[i,j] = (math.log(1+img[i,j])) 
				cv2.normalize(img2,img2,0,255,cv2.NORM_MINMAX)
				return cv2.convertScaleAbs(img2)	
		
		def gamma_tran(img, gamma): #gamma=0.05~5
			# 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
			gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
			gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
			
			# 实现这个映射用的是OpenCV的查表函数
			return cv2.LUT(img, gamma_table)
		
		img1 = img.copy()
		if 0==type:
			return equalizeHist(img1)
		elif 1==type:
			return laplace(img1)
		elif 2==type:
			return log(img1)
		elif 3==type:
			gamma = getImgMean(img1)
			gamma = (5-0.05)*(gamma/255)
			#print gamma
			return gamma_tran(img1, gamma) #0.05-5
		
		
	def invertColor(self,img, min_h, max_h, type=0):
		def changeHue(img, min_h, max_h): # change image color by hsv
			if 3==len(img.shape):
				img = img.astype(np.uint8)
				hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				_h = uint16(random.randint(min_h, max_h))
				hsv_img[:, :, 0] += _h
				out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
				return out
			else:
				return None
		
		img1 = img.copy()
		if 0==type:
			return changeHue(img1, min_h, max_h)
		
	
	def zoom(self,img, p1x, p1y, p2x, p2y):
		h = img.shape[0]
		w = img.shape[1]

		crop_p1x = max(p1x, 0)
		crop_p1y = max(p1y, 0)
		crop_p2x = min(p2x, w-1)
		crop_p2y = min(p2y, h-1)

		cropped_img = img[crop_p1y:crop_p2y, crop_p1x:crop_p2x]

		x_pad_before = -min(0, p1x)
		x_pad_after	 =	max(0, p2x-w)
		y_pad_before = -min(0, p1y)
		y_pad_after	 =	max(0, p2y-h)

		padding = [(y_pad_before, y_pad_after), (x_pad_before, x_pad_after)]
		is_colour = len(img.shape) == 3
		if is_colour:
			padding.append((0,0)) # colour images have an extra dimension

		padded_img = np.pad(cropped_img, padding, 'constant')
		#print padded_img.shape
		#print "11111111"
		return cv2.resize(padded_img, (w,h))
	
	def translate(self, img, x_trans, y_trans, type=0):
		height = img.shape[0]
		width = img.shape[1]
		if x_trans>width or y_trans>height:
			print "error:x_trans>width or y_trans>height"
			return None
			
		if 0==type:
		 # 定义平移矩阵
			M = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
			return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
		else:
			length = len(img.shape)
			if 3==length:
				img2 = np.zeros((height-y_trans,width-x_trans,3))
				for i in range(height-y_trans):
					for j in range(width-x_trans):
						#print i,j, img2.shape
						img2[i,j] = (img[i,j][0],img[i,j][1],img[i,j][2]) 
				return img2	
			else:
				img2 = np.zeros((height-y_trans,width-x_trans))
				for i in range(height-y_trans):
					for j in range(width-x_trans):
						img2[i,j] = (img[i,j]) 
				return img2	
		
	def rotate(self, img, angle, scale=1.0, type=0):
		def rotate_change(img, angle): #旋转不带放大
			"""Extract a rectangle from the source image.
			
			image - source image
			center - (x,y) tuple for the centre point.
			theta - angle of rectangle.
			width, height - rectangle dimensions.
			"""
			height = img.shape[0]  
			width = img.shape[1] 
			centerX	 = width/2
			centerY = height/2
			#print img.shape
			
			angle = np.deg2rad(angle)	
			#theta *= math.pi / 180 # convert to rad
			v_x = (math.cos(angle), math.sin(angle))
			v_y = (-math.sin(angle), math.cos(angle))
			s_x = centerX - v_x[0] * (width / 2) - v_y[0] * (height / 2)
			s_y = centerY - v_x[1] * (width / 2) - v_y[1] * (height / 2)
			mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])
			#mapping = np.array([[v_x[0],v_y[0],0], [v_x[1],v_y[1],0]])
			
			s_x = v_x[0] * (width / 2) + v_y[0] * (height / 2)
			s_y = v_x[1] * (width / 2) + v_y[1] * (height / 2)
			
			return cv2.warpAffine(img, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
			
		'''def rotate_nochange(src, angle, scale=1.):#旋转带放大，图像信息没丢失
			w = src.shape[1]
			h = src.shape[0]
			rangle = -np.deg2rad(angle)	 # angle in radians
			# now calculate new image width and height
			nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
			nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
			# ask OpenCV for the rotation matrix
			rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
			# calculate the move from the old center to the new center combined
			# with the rotation
			rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
			# the move only affects the translation, so update the translation
			# part of the transform
			rot_mat[0,2] += rot_move[0]
			rot_mat[1,2] += rot_move[1]
			return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))))
			
		img1 = img.copy()
		if 0==type:
			return rotate_change(img1, angle)
		else:
			return rotate_nochange(img1, angle, scale)'''
		def rotate_nochange(src, angle, scale=1.):#旋转带放大，图像信息没丢失
			w = src.shape[1]
			h = src.shape[0]
			rangle = -np.deg2rad(angle)	 # angle in radians
			rot_mat = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
			return cv2.warpAffine(src, rot_mat, (w, h))
			
		img1 = img.copy()
		if 0==type:
			return rotate_change(img1, angle)
		else:
			return rotate_nochange(img1, angle, scale)
	def noise(self, img, param=0.1, type=0):
		
		def SaltAndPepper(src, percetage): #定义添加椒盐噪声的函数 
			SP_NoiseImg=src 
			SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
			for i in range(SP_NoiseNum): 
				randX=random.random_integers(0,src.shape[0]-1) 
				randY=random.random_integers(0,src.shape[1]-1) 
				if random.random_integers(0,1)==0: 
					SP_NoiseImg[randX,randY]=0 
				else: 
					SP_NoiseImg[randX,randY]=255 
			return SP_NoiseImg
			
		def gaussain_noise(img, mean = 0, var = 0.1) :
			img = img.astype(np.uint8)
			
			sigma = var ** 0.5
			gauss = np.random.normal(mean, sigma, img.shape)
			gauss = gauss.reshape(img.shape).astype(np.uint8)
			noisy = img + gauss
			return noisy

		img1 = img.copy()
		if 0==type:
			return gaussain_noise(img1, mean=0, var=param)
		else:
			return SaltAndPepper(img1, param)

	def flip(self, img, flipCode):
		return cv2.flip(img,flipCode)
		'''if isH:
			return np.fliplr(img) #上下
		else:
			return np.flipud(img)'''
	
	def blur(self, img, kernel_dim=5, sigma=1.5, type=0):
		if 0==type:
			result = cv2.blur(img, (kernel_dim,kernel_dim))
		elif 1==type:
			result = cv2.GaussianBlur(img,(kernel_dim,kernel_dim),sigma)  
		else:
			result = cv2.medianBlur(img,kernel_dim) 
		return result
		
	
	#源代码是针对正方形的图像，所以在对输入任意大小的图像采用crop进行处理，最后在还原成原图大小
	def elastic(self, img, kernel_dim=31, sigma=6, alpha=47):
		length = len(img.shape)
		w = img.shape[1]
		h = img.shape[0]
		img2 = img.copy()
		#print img.shape
		
		if 3==length:
			img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
			
		if w>h:
			img1 = np.zeros((w,w))
			img1[0:h][:] = img2[:][:]
			distorted_image = elastic_distortion(img1, kernel_dim, sigma, alpha)
			img_crop = distorted_image[0:h,:]
			return img_crop
		else:
			img1 = np.zeros((h,h))
			#img1[:][0:w] = img2[:][:] #会报错，所以使用下面的代码
			for i in range(img2.shape[0]):
				for j in range(img2.shape[1]):
					img1[i,j] = img2[i,j]
	
			distorted_image = elastic_distortion(img1, kernel_dim, sigma, alpha)
			img_crop = distorted_image[:,0:w]
			return img_crop
		
	def inverse(self, img):
		length = len(img.shape)
		height = img.shape[0]
		width = img.shape[1]
		
		img2 = img.copy()
		
		if 3==length:
			for i in range(height):
				for j in range(width):
					img2[i,j] = (255-img[i,j][0],255-img[i,j][1],255-img[i,j][2]) 
		else:
			for i in range(height):
				for j in range(width):
					img2[i,j] = (255-img[i,j]) 
		return img2

##########################################################################################
#计算坐标变换的相关代码
def get_zoom(img, result, p1x, p1y, p2x, p2y, xmin,ymin, xmax, ymax):
	xmin1 = []
	ymin1 = []
	xmax1 = []
	ymax1 = []
	w = img.shape[1]
	h = img.shape[0]
	pw = p2x - p1x
	ph = p2y - p1y
	#print pw,ph
		
	for i in range(len(xmin)):
		if xmin[i] < p1x:
			x1 = 0
		else:
			x1 = xmin[i] - p1x
			
		if ymin[i] < p1y:
			y1 = 0
		else:
			y1 = ymin[i] - p1y
			
		if xmax[i] < p2x:
			x2 = xmax[i] - p1x
		else:
			x2 = pw-1
			
		if ymax[i] < p2y:
			y2 = ymax[i] - p1y
		else:
			y2 = ph-1
			
		#print x1,y1,x2,y2
		
		x1 = x1*w/pw
		x2 = x2*w/pw
		y1 = y1*h/ph
		y2 = y2*h/ph
		#print x1,y1,x2,y2
		xmin1.append ( x1 )
		ymin1.append ( y1 )
		xmax1.append ( x2 )
		ymax1.append ( y2 )
		#cv2.rectangle (result,(xmin1[i],ymin1[i]),(xmax1[i],ymax1[i]),(0,0,255),1)
	#cv2.imwrite("test1.jpg", result)
	return xmin1,ymin1,xmax1,ymax1
	
def get_flip(img, flipCode, xmin,ymin, xmax, ymax):
	xmin1 = []
	ymin1 = []
	xmax1 = []
	ymax1 = []
	w = img.shape[1]
	h = img.shape[0]
	if flipCode>0:
		for i in range(len(xmin)):
			xmin1.append ( w - 1 - xmax[i] )
			ymin1.append ( ymin[i] )
			xmax1.append ( w - 1 - xmin[i] )
			ymax1.append ( ymax[i] )
			#cv2.rectangle (img1,(xmin1[i],ymin1[i]),(xmax1[i],ymax1[i]),(0,0,255),1)
				
	if 0==flipCode:
		for i in range(len(xmin)):
			xmin1.append ( xmin[i] )
			ymin1.append ( h - 1 - ymax[i] )
			xmax1.append ( xmax[i] )
			ymax1.append ( h - 1 - ymin[i] )
						
	if flipCode<0:
		for i in range(len(xmin)):
			xmin1.append ( w - 1 - xmax[i] )
			ymin1.append ( h - 1 - ymax[i] )
			xmax1.append ( w - 1 - xmin[i] )
			ymax1.append ( h - 1 - ymin[i] )
			#cv2.rectangle (result,(xmin1[i],ymin1[i]),(xmax1[i],ymax1[i]),(0,0,255),1)
		#cv2.imwrite("test1.jpg", result)
	return xmin1,ymin1,xmax1,ymax1
	
def get_translate(img, x_trans, y_trans, xmin,ymin, xmax, ymax):
	xmin1 = []
	ymin1 = []
	xmax1 = []
	ymax1 = []
	w = img.shape[1]
	h = img.shape[0]
	w = w - x_trans
	h = h - y_trans
	for i in range(len(xmin)):
		x1 = w-1 if xmin[i]>=w else xmin[i]  #坐标不可能为负数，所以不考虑这种情况
		x2 = w-1 if xmax[i]>=w else xmax[i]
		y1 = h-1 if ymin[i]>=h else ymin[i]
		y2 = h-1 if ymax[i]>=h else ymax[i]
		xmin1.append ( x1 )
		ymin1.append ( y1 )
		xmax1.append ( x2 )
		ymax1.append ( y2 )
	return xmin1,ymin1,xmax1,ymax1
	
def rotate_XY(x,y, angle):
	angle = -angle #
	x1 = int(x*np.cos(angle) + y*np.sin(angle))
	y1 = int(-x*np.sin(angle) + y*np.cos(angle))
	return int(x1), int(y1)

def get_offset(x,y,w,h,angle, centerX1, centerY1):
	centerX0 = x + w/2
	centerY0 = y + h/2
	
	dx = int(centerX1 - centerX0)
	dy = int(centerY1 - centerY0)
	return dx,dy

def get_center(x1,y1,x2,y2,x3,y3,x4,y4):
	x_min = min(x1,x2,x3,x4)
	x_max = max(x1,x2,x3,x4)
	y_min = min(y1,y2,y3,y4)
	y_max= max(y1,y2,y3,y4)
	(x,y) = (int((x_min + x_max)) / 2.0, int((y_min + y_max) / 2.0))
	return x,y

def get_rotate_change(img, result, angle, xmin, ymin, xmax,ymax):
	rad = np.deg2rad(angle)
	xmin1 = []
	ymin1 = []
	xmax1 = []
	ymax1 = []
	width = img.shape[1]
	height = img.shape[0]
	centerX = img.shape[1]/2
	centerY = img.shape[0]/2
	#print img.shape
		
	angle = np.deg2rad(angle)
	v_x = (math.cos(rad), math.sin(rad))
	v_y = (-math.sin(rad), math.cos(rad))
	s_x = centerX - v_x[0] * (width / 2) - v_y[0] * (height / 2)
	s_y = centerY - v_x[1] * (width / 2) - v_y[1] * (height / 2)
	#print img.shape
	#print result.shape
	result1=result.copy()
	result2=result.copy()    
	for i in range(len(xmin)):
		
		
		'''xmin[i] = int (xmin[i] - s_x)
		xmax[i] = int (xmax[i] - s_x)
		ymin[i] = int (ymin[i] - s_y)
		ymax[i] = int (ymax[i] - s_y)'''
		
		x1,y1 = rotate_XY(xmin[i],ymin[i],-rad)
		x2,y2 = rotate_XY(xmax[i],ymin[i],-rad)
		x3,y3 = rotate_XY(xmax[i],ymax[i],-rad)
		x4,y4 = rotate_XY(xmin[i],ymax[i],-rad)
		#centerX1, centerY1 = get_center(x1,y1,x2,y2,x3,y3,x4,y4)
		#print x1,y1,x3,y3
		#dx,dy = get_offset(xmin[i],ymin[i],xmax[i]-xmin[i],ymax[i]-ymin[i],rad, centerX1, centerY1)
						
		#dx = dx -int (result.shape[1]/2 - img.shape[1]/2)
		#dy = dy -int (result.shape[0]/2 - img.shape[0]/2)

		#cv2.rectangle(result1,(x_min,y_min),(x_max,y_max),(0,0,255),1)	
		#cv2.rectangle(result1,(xmin[i],ymin[i]),(xmax[i],ymax[i]),(255,255,255),1)	
						
		'''x1 = int (x1 - dx)
		x2 = int (x2 - dx)
		x3 = int (x3 - dx)
		x4 = int (x4 - dx)
		y1 = int (y1 - dy)
		y2 = int (y2 - dy)
		y3 = int (y3 - dy)
		y4 = int (y4 - dy)'''  
		x1 = int (x1 - s_x)
		x2 = int (x2 - s_x)
		x3 = int (x3 - s_x)
		x4 = int (x4 - s_x)
		y1 = int (y1 - s_y)
		y2 = int (y2 - s_y)
		y3 = int (y3 - s_y)
		y4 = int (y4 - s_y)  

       
						
		x_min = int(min(x1,x2,x3,x4))
		x_max = int(max(x1,x2,x3,x4))
		y_min = int(min(y1,y2,y3,y4))
		y_max = int(max(y1,y2,y3,y4))
			
		'''xmin1.append ( x_min )
		ymin1.append ( y_min )
		xmax1.append ( x_max )
		ymax1.append ( y_max )'''
		'''cv2.rectangle (result,(x_min,y_min),(x_max,y_max),(0,255,255),5)
		cv2.line (result,(x1,y1),(x2,y2),255,2)
		cv2.line (result,(x2,y2),(x3,y3),255,2)
		cv2.line (result,(x3,y3),(x4,y4),255,2)
		cv2.line (result,(x4,y4),(x1,y1),255,2)'''
		#print x_min,y_min,x_max,y_max
		x_min = max(x_min,1)
		y_min = max(y_min,1)
		x_max = min(x_max, img.shape[1]-2)
		y_max = min(y_max, img.shape[0]-2)
		#print x_min,y_min,x_max,y_max
		xmin1.append( x_min )
		ymin1.append( y_min )
		xmax1.append( x_max )
		ymax1.append( y_max )		
		cv2.rectangle (result1,(xmin1[i],ymin1[i]),(xmax1[i],ymax1[i]),(255,255,0),1)			
	cv2.imwrite("test1.jpg", result1)
	#cv2.imwrite("test2.jpg", result2)
	return xmin1,ymin1,xmax1,ymax1
	
def get_rotate_nochange(img, result, angle, xmin, ymin, xmax,ymax):
	rad = -np.deg2rad(angle)
	xmin1 = []
	ymin1 = []
	xmax1 = []
	ymax1 = []
	w = img.shape[1]
	h = img.shape[0]
	rot_mat = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
	print(type(rot_mat))

	result1=result.copy()
	#print img.shape
	#print result.shape
	for i in range(len(xmin)):
		#cv2.rectangle (result1,(xmin[i],ymin[i]),(xmax[i],ymax[i]),(255,255,255),2)
		tl = np.dot(rot_mat,np.expand_dims(np.array([xmin[i],ymin[i],1]),axis=1))
		x1=tl[0]
		y1=tl[1]
		tr = np.dot(rot_mat,np.expand_dims(np.array([xmax[i],ymin[i],1]),axis=1))
		x2=tr[0]
		y2=tr[1]
		br = np.dot(rot_mat,np.expand_dims(np.array([xmax[i],ymax[i],1]),axis=1))
		x3=br[0]
		y3=br[1]
		bl = np.dot(rot_mat,np.expand_dims(np.array([xmin[i],ymax[i],1]),axis=1))
		x4=bl[0]
		y4=bl[1]

						
		x_min = int(min(x1,x2,x3,x4))
		x_max = int(max(x1,x2,x3,x4))
		y_min = int(min(y1,y2,y3,y4))
		y_max = int(max(y1,y2,y3,y4))

		'''cv2.rectangle (result,(x_min,y_min),(x_max,y_max),(0,255,255),5)
		cv2.line (result,(x1,y1),(x2,y2),255,2)
		cv2.line (result,(x2,y2),(x3,y3),255,2)
		cv2.line (result,(x3,y3),(x4,y4),255,2)
		cv2.line (result,(x4,y4),(x1,y1),255,2)'''
		xmin1.append( x_min )
		ymin1.append( y_min )
		xmax1.append( x_max )
		ymax1.append( y_max )
		cv2.rectangle (result1,(xmin1[i],ymin1[i]),(xmax1[i],ymax1[i]),(0,0,255),2)			
	cv2.imwrite("test1.jpg", result1)
	return xmin1,ymin1,xmax1,ymax1
'''def get_rotate_nochange(img, result, angle, xmin, ymin, xmax,ymax):
	rad = -np.deg2rad(angle)
	xmin1 = []
	ymin1 = []
	xmax1 = []
	ymax1 = []
	#print img.shape
	#print result.shape
	for i in range(len(xmin)):
		x1,y1 = rotate_XY(xmin[i],ymin[i],rad)
		x2,y2 = rotate_XY(xmax[i],ymin[i],rad)
		x3,y3 = rotate_XY(xmax[i],ymax[i],rad)
		x4,y4 = rotate_XY(xmin[i],ymax[i],rad)
		centerX1, centerY1 = get_center(x1,y1,x2,y2,x3,y3,x4,y4)
		#print x1,y1,x3,y3
		dx,dy = get_offset(xmin[i],ymin[i],xmax[i]-xmin[i],ymax[i]-ymin[i],rad, centerX1, centerY1)
						
		dx = dx -int (result.shape[1]/2 - img.shape[1]/2)
		dy = dy -int (result.shape[0]/2 - img.shape[0]/2)
						
		x1 = int (x1 - dx)
		x2 = int (x2 - dx)
		x3 = int (x3 - dx)
		x4 = int (x4 - dx)
		y1 = int (y1 - dy)
		y2 = int (y2 - dy)
		y3 = int (y3 - dy)
		y4 = int (y4 - dy)
						
		x_min = int(min(x1,x2,x3,x4))
		x_max = int(max(x1,x2,x3,x4))
		y_min = int(min(y1,y2,y3,y4))
		y_max = int(max(y1,y2,y3,y4))
			
		xmin1.append ( x_min )
		ymin1.append ( y_min )
		xmax1.append ( x_max )
		ymax1.append ( y_max )
		#cv2.rectangle (result,(x_min,y_min),(x_max,y_max),(0,255,255),5)
		#cv2.line (result,(x1,y1),(x2,y2),255,2)
		#cv2.line (result,(x2,y2),(x3,y3),255,2)
		#cv2.line (result,(x3,y3),(x4,y4),255,2)
		#cv2.line (result,(x4,y4),(x1,y1),255,2)
		xmin1.append( x_min )
		ymin1.append( y_min )
		xmax1.append( x_max )
		ymax1.append( y_max )
		#cv2.rectangle (result,(xmin1[i],ymin1[i]),(xmax1[i],ymax1[i]),(0,0,255),2)			
	#cv2.imwrite("test1.jpg", result)
	return xmin1,ymin1,xmax1,ymax1'''
def read_xml( path ):
	#打开xml文档
	dom = xml.dom.minidom.parse( path ) #用于打开一个xml文件，并将这个文件对象dom变量。
	root = dom.documentElement #用于得到dom对象的文档元素，并把获得的对象给root
	xmin_lst = root.getElementsByTagName("xmin")
	ymin_lst = root.getElementsByTagName("ymin")
	xmax_lst = root.getElementsByTagName("xmax")
	ymax_lst = root.getElementsByTagName("ymax")
	
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	
	for i in range(len(xmin_lst)):
		xmin.append( int(xmin_lst[i].firstChild.data) )
		ymin.append( int(ymin_lst[i].firstChild.data) )
		xmax.append( int(xmax_lst[i].firstChild.data) )
		ymax.append( int(ymax_lst[i].firstChild.data) )
	return xmin,ymin,xmax,ymax

def write_xml( path, save_path, xmin,ymin,xmax,ymax, image_width, image_height):
	#读
	dom = xml.dom.minidom.parse( path ) #用于打开一个xml文件，并将这个文件对象dom变量。
	root = dom.documentElement #用于得到dom对象的文档元素，并把获得的对象给root
	
	folder = root.getElementsByTagName("folder")
	filename = root.getElementsByTagName("filename")

	database = root.getElementsByTagName("database")
	annotation = root.getElementsByTagName("annotation")
	image = root.getElementsByTagName("image")
	flickrid = root.getElementsByTagName("flickrid")

	name = root.getElementsByTagName("name")

	width = root.getElementsByTagName("width")
	height = root.getElementsByTagName("height")
	depth = root.getElementsByTagName("depth")

	object = root.getElementsByTagName("object")
	pose = root.getElementsByTagName("pose")
	truncated = root.getElementsByTagName("truncated")
	difficult = root.getElementsByTagName("difficult")
	bndbox = root.getElementsByTagName("bndbox")
	
	#写
	impl = xml.dom.minidom.getDOMImplementation() 
	dom = impl.createDocument(None, 'annotation' , None) 
	root = dom.documentElement  
	nameE = dom.createElement( 'folder' ) 
	nameT = dom.createTextNode( folder[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	root.appendChild(nameE) 
	nameE = dom.createElement( 'filename' ) 
	nameT = dom.createTextNode( filename[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	root.appendChild(nameE) 

	source = dom.createElement( 'source' ) 
	root.appendChild(source) 
	nameE = dom.createElement( 'database' ) 
	nameT = dom.createTextNode( database[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	source.appendChild(nameE) 
	nameE = dom.createElement( 'annotation' ) 
	nameT = dom.createTextNode( annotation[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	source.appendChild(nameE) 
	nameE = dom.createElement( 'image' ) 
	nameT = dom.createTextNode( image[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	source.appendChild(nameE) 
	nameE = dom.createElement( 'flickrid' ) 
	nameT = dom.createTextNode( flickrid[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	source.appendChild(nameE) 

	owner = dom.createElement( 'owner' ) 
	root.appendChild(owner) 
	nameE = dom.createElement( 'flickrid' ) 
	nameT = dom.createTextNode( flickrid[1].firstChild.data ) 
	nameE.appendChild(nameT) 
	owner.appendChild(nameE) 
	nameE = dom.createElement( 'name' ) 
	nameT = dom.createTextNode( name[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	owner.appendChild(nameE) 

	size = dom.createElement( 'size' ) 
	root.appendChild(size) 
	nameE = dom.createElement( 'width' ) 
	nameT = dom.createTextNode( str(image_width) )#width[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	size.appendChild(nameE) 
	nameE = dom.createElement( 'height' ) 
	nameT = dom.createTextNode( str(image_height) )#height[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	size.appendChild(nameE) 
	nameE = dom.createElement( 'depth' ) 
	nameT = dom.createTextNode( depth[0].firstChild.data ) 
	nameE.appendChild(nameT) 
	size.appendChild(nameE) 

	segmented = dom.createElement( 'segmented' )
	root.appendChild(segmented)
	
	for i in range(len(object)):
		object = dom.createElement( 'object' ) 
		root.appendChild(object) 
		nameE = dom.createElement( 'name' ) 
		nameT = dom.createTextNode( name[i+1].firstChild.data ) 
		nameE.appendChild(nameT) 
		object.appendChild(nameE) 
		nameE = dom.createElement( 'pose' ) 
		nameT = dom.createTextNode( pose[i].firstChild.data ) 
		nameE.appendChild(nameT) 
		object.appendChild(nameE) 
		nameE = dom.createElement( 'truncated' ) 
		nameT = dom.createTextNode( truncated[i].firstChild.data ) 
		nameE.appendChild(nameT) 
		object.appendChild(nameE) 
		nameE = dom.createElement( 'difficult' ) 
		nameT = dom.createTextNode( difficult[i].firstChild.data ) 
		nameE.appendChild(nameT) 
		object.appendChild(nameE) 

		bndbox = dom.createElement( 'bndbox' ) 
		object.appendChild(bndbox) 
		nameE = dom.createElement( 'xmin' ) 
		nameT = dom.createTextNode( str(xmin[i]) ) 
		nameE.appendChild(nameT) 
		bndbox.appendChild(nameE) 
		nameE = dom.createElement( 'ymin' ) 
		nameT = dom.createTextNode( str(ymin[i]) ) 
		nameE.appendChild(nameT) 
		bndbox.appendChild(nameE) 
		nameE = dom.createElement( 'xmax' ) 
		nameT = dom.createTextNode( str(xmax[i]) ) 
		nameE.appendChild(nameT) 
		bndbox.appendChild(nameE) 
		nameE = dom.createElement( 'ymax' ) 
		nameT = dom.createTextNode( str(ymax[i]) ) 
		nameE.appendChild(nameT) 
		bndbox.appendChild(nameE) 
		
	f = open( save_path , 'w') 
	dom.writexml(f, addindent = ' ' , newl = '\n' ,encoding = 'utf-8' )
	f.close()  