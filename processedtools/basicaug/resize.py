import os
from PIL import Image
 
filename = os.listdir("D:\\Work\\process\\样本处理\\polyu-all-train")
base_dir = "D:\\Work\\process\\样本处理\\polyu-all-train\\"
new_dir  = "D:\\Work\\process\\样本处理\\polyu\\"
size_m = 128
size_n = 128
 
for img in filename:
    image = Image.open(base_dir + img)
    image_size = image.resize((size_m, size_n),Image.ANTIALIAS)
    image_size.save(new_dir+ img)