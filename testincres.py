from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import inception_resnet_v1
from input_data import get_files

# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    image = np.array(img)
    return image


# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 128,128, 3])

        logit = inception_resnet_v1.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[128, 128, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'F:/kidneycaRunDemon-Akimoto-manazu/save'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.inception_resnet_v1_checkpoint_path:
                global_step = ckpt.inception_resnet_v1_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.inception_resnet_v1_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                result = ('这是RCC的可能性为： %.6f' % prediction[:, 0])
            elif max_index == 1:
                result = ('这是papillary的可能性为： %.6f' % prediction[:, 1])
            elif max_index == 2:
              result = ('这是others的可能性为： %.6f' % prediction[:, 2])
            #else:
             #   result = ('这是这是向日葵的可能性为： %.6f' % prediction[:, 3])
            return result


# ------------------------------------------------------------------------

if __name__ == '__main__':
    img = Image.open('F:/kidneycaRunDemon-Akimoto-manazu/inputdata/12.png')
    plt.imshow(img)
    plt.show()
    imag = img.resize([128, 128])
    image = np.array(imag)
    evaluate_one_image(image)
