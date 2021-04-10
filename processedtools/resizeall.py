import os
from PIL import Image
 
filename1 = os.listdir("/home/zyh/下载/WeiboAlbum/清洗后/kubo/")
base_dir1 = "/home/zyh/下载/WeiboAlbum/清洗后/kubo/"
new_dir1  = "/home/zyh/下载/WeiboAlbum/resize/kubo/"
size_m = 128
size_n = 128
 
for img in filename1:
    image = Image.open(base_dir1 + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(new_dir1+ img)
    print("resize kubo done")

filename2 = os.listdir("/home/zyh/下载/WeiboAlbum/清洗后/serei/")
base_dir2 = "/home/zyh/下载/WeiboAlbum/清洗后/serei/"
new_dir2  = "/home/zyh/下载/WeiboAlbum/resize/serei/"
size_m = 128
size_n = 128
 
for img in filename2:
    image = Image.open(base_dir2 + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(new_dir2+ img)
    print("resize serei done")

filename3 = os.listdir("/home/zyh/下载/WeiboAlbum/清洗后/mayu/")
base_dir3 = "/home/zyh/下载/WeiboAlbum/清洗后/mayu/"
new_dir3  = "/home/zyh/下载/WeiboAlbum/resize/mayu/"
size_m = 128
size_n = 128
 
for img in filename3:
    image = Image.open(base_dir3 + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(new_dir3+ img)
    print("resize mayu done")

filename4 = os.listdir("/home/zyh/下载/WeiboAlbum/清洗后/miu/")
base_dir4 = "/home/zyh/下载/WeiboAlbum/清洗后/miu/"
new_dir4  = "/home/zyh/下载/WeiboAlbum/resize/miu/"
size_m = 128
size_n = 128
 
for img in filename4:
    image = Image.open(base_dir4 + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(new_dir4+ img)
    print("resize miu done")

filename5 = os.listdir("/home/zyh/下载/WeiboAlbum/清洗后/notabove/")
base_dir5 = "/home/zyh/下载/WeiboAlbum/清洗后/notabove/"
new_dir5  = "/home/zyh/下载/WeiboAlbum/resize/notabove/"
size_m = 128
size_n = 128
 
for img in filename5:
    image = Image.open(base_dir5 + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(new_dir5+ img)
    print("resize notabove done")
