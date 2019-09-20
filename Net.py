import tensorflow as tf
import numpy as np
import PIL.Image as img
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
from sample_train1 import Sample
from sample_validate1 import Sample1

class CNN_net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,5])
        self.w1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,64],dtype=tf.float32,stddev=np.sqrt(1/64)))
        self.b1 = tf.Variable(tf.zeros(shape=[64],dtype=tf.float32))#224*224
        self.w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=np.sqrt(1 / 128)))
        self.b2 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))#112*112
        self.w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], dtype=tf.float32, stddev=np.sqrt(1 / 256)))
        self.b3 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))#56*56
        self.w4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], dtype=tf.float32, stddev=np.sqrt(1 / 256)))
        self.b4 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))#28*28
        self.w5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], dtype=tf.float32, stddev=np.sqrt(1 / 512)))
        self.b5 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))#14*14
        self.w6 = tf.Variable(tf.truncated_normal(shape=[7*7*512, 512], dtype=tf.float32, stddev=np.sqrt(1 / 512)))
        self.b6 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))#7*7
        self.w7 = tf.Variable(tf.truncated_normal(shape=[512, 256], dtype=tf.float32, stddev=np.sqrt(1 / 256)))
        self.b7 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))
        self.w8 = tf.Variable(tf.truncated_normal(shape=[256, 5], dtype=tf.float32, stddev=np.sqrt(1 / 5)))
        self.b8 = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32))
    def farward(self):
        y1 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(self.x,self.w1,strides=[1,1,1,1],padding='SAME')+self.b1)))#224*224*64
        pool_y1 = tf.nn.max_pool(y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#112*112*64
        # print(pool_y1.shape)
        y2 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(pool_y1,self.w2,strides=[1,1,1,1],padding='SAME')+self.b2)))  # 112*112*128
        pool_y2= tf.nn.max_pool(y2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#56*56*128
        # print(pool_y2.shape)
        y3 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(pool_y2,self.w3, strides=[1,1,1,1],padding='SAME')+self.b3)))  # 56*56*256
        pool_y3 = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 28*28*256
        # print(pool_y3.shape)
        y4 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(pool_y3,self.w4, strides=[1,1,1,1], padding='SAME') + self.b4)))  # 28*28*256
        pool_y4 = tf.nn.max_pool(y4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 14*14*256
        # print(pool_y4.shape)
        y5 = tf.nn.relu(tf.layers.batch_normalization((tf.nn.conv2d(pool_y4, self.w5, strides=[1, 1, 1, 1], padding='SAME') + self.b5)))  # 14*14*512
        pool_y5 = tf.nn.max_pool(y5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 7*7*512
        # print(pool_y5.shape)
        pool_y5 = tf.reshape(pool_y5,[-1,7*7*512])
        # print(pool_y5.shape)#-1,7*7*512

        y6 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(pool_y5,self.w6)+self.b6))
        y7 =tf.nn.relu(tf.layers.batch_normalization(tf.matmul(y6,self.w7)+self.b7))
        self.y8 = tf.layers.batch_normalization(tf.matmul(y7,self.w8)+self.b8)
        self.output = self.y8
    def loss(self):
        self.label_coord = self.y[:,:4]  # [10,4]
        self.label_conf = self.y[:,4:]  # [10,1]

        self.coord = self.output[:,:4]  # [10,4]
        self.conf_loss = self.output[:,4:]  # [10,1]

        self.conf_out = tf.nn.sigmoid(self.conf_loss)
        self.coord_loss = tf.reduce_mean((self.label_coord-self.coord)**2)
        self.conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_conf,logits=self.conf_loss))
        self.tatol_loss = self.coord_loss+self.conf_loss
    def backward(self):
        self.optimizer = tf.train.AdamOptimizer().minimize(self.tatol_loss)
cnn = CNN_net()
cnn.farward()
cnn.loss()
cnn.backward()
if __name__ == '__main__':
    # init = tf.global_variables_initializer()
    save =tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        save.restore(sess,'./my_net/save_net.ckpt')
        sample = Sample()
        sample.read_data()
        sample1 = Sample()
        sample1.read_data()
        j =1
        for i in range(10000):
            xs,ys =sample.get_batch(5)
            error,_=sess.run(fetches=[cnn.tatol_loss,cnn.optimizer],feed_dict={cnn.x:xs,cnn.y:ys})
            if i %3 ==0:
                xss,yss = sample1.get_batch(2)
                # print(yss)
                test_code,conf,_error,coord_error,conf_error,out = sess.run(
                    fetches=[cnn.coord,cnn.conf_out,cnn.tatol_loss,cnn.coord_loss,cnn.conf_loss,cnn.output],
                    feed_dict={cnn.x:xss,cnn.y:yss})
                print('训练次数：', j)
                print("训练损失:", error, "验证损失：", _error,coord_error,conf_error)

                x1 = test_code[0][0]*224
                y1 = test_code[0][1]*224
                x2 = test_code[0][2]*224
                y2 = test_code[0][3]*224
                test_confidence = conf[0][0]
                #
                img1 = (np.array(xss) / 2 + 0.5) * 255
                img2 = img.fromarray(np.uint8(img1[0]), 'RGB')
                validate_x1 = yss[0][0]*224
                validate_y1 = yss[0][1]*224
                validate_x2 = yss[0][2]*224
                validate_y2 = yss[0][3]*224

                print('label:',validate_x1,validate_y1,validate_x2,validate_y2)
                print('output:',x1,y1,x2,y2)
                print('confidence:',test_confidence)
                #
                img3 = draw.Draw(img2)
                img3.rectangle(xy=[x1,y1,x2,y2],outline='blue')
                img3.rectangle(xy=[validate_x1, validate_y1, validate_x2, validate_y2], outline='red')
                # img2.show()
                plt.imshow(img2)
                plt.pause(1)
                j+=1

                save.save(sess,"./my_net/save_net.ckpt")




