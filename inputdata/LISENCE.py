 
#用于在制作cifar10数据集时输出lst文件
import os
import os.path

rootdir1 = "/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/kidney_data_mri1/kidney_photos/chome"
rootdir2 = "/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/kidney_data_mri1/kidney_photos/papillary"
rootdir3 = "/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/kidney_data_mri1/kidney_photos/notca"
file_object = open('/home/zyh/桌面/kidneycaRunDemon-Shiraishi-Mai/inputdata/kidney_data_mri1/kidney_photos/Lisence.txt','w')
 
for parent,dirnames,filenames in os.walk(rootdir1):
	for filename in filenames:
		print(filename)
		if ".png" in filename:
                 file_object.write('chome/' + filename + '\n')

for parent,dirnames,filenames in os.walk(rootdir2):
	for filename in filenames:
		print(filename)
		if ".png" in filename:
                 file_object.write('papillary/'+filename + '\n')

 
for parent,dirnames,filenames in os.walk(rootdir3):
	for filename in filenames:
		print(filename)
		if ".png" in filename:
                 file_object.write('notca/'+filename + '\n')
file_object.close()

