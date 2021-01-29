# 导入文件
import os
import numpy as np
import tensorflow as tf
import input_data
import densenet

# 变量声明
N_CLASSES = 3 # 分类数
IMG_W = 512  # resize图像，太大的话训练时间久
IMG_H = 512
BATCH_SIZE = 20
CAPACITY = 520
MAX_STEP = 700 # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001

# 获取批次batch
train_dir = '/content/gdrive/My Drive/twokidneyca/inputdata'  # 训练样本的读入路径
logs_train_dir = '/content/gdrive/My Drive/twokidneyca/save'  # logs存储路径

# train, train_label = input_data.get_files(train_dir)
train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)
# 训练数据及标签
train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 测试数据及标签
val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 训练操作定义
train_logits = densenet.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = densenet.losses(train_logits, train_label_batch)
train_op = densenet.trainning(train_loss, learning_rate)
train_acc = densenet.evaluation(train_logits, train_label_batch)

# 测试操作定义
test_logits = densenet.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = densenet.losses(test_logits, val_label_batch)
test_acc = densenet.evaluation(test_logits, val_label_batch)

# 这个是log汇总记录
summary_op = tf.summary.merge_all()
