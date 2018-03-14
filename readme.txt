function.py : 增强函数、检测时坐标变换的函数都在该文件下
test_detect.py ： 检测数据集增强
test_recognition.py： 识别数据集增强
synthesis.py： 增强后图像、xml(txt)合并


img_detect:			检测数据集的原始数据
result_detect:		检测数据集增强后的

img_recognition:	识别数据集的原始数据
result_recognition:	识别数据集增强后的

all：合并的文件夹，根据合并是检测还是识别里面内容有所不同，
	主要是合并result_detect或result_recognition
	
要想将原始数据、增强数据一起合并：
	cat 1.txt 2.txt > 3.txt  #将1.txt 2.txt的文件内容合并到3.txt
	检测数据集：
		cp src/xml/*  dst/xml/
		cp src/img/*  dst/img/
	识别数据集：
		cp src/*  dst/
