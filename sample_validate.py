import numpy as np
import os
import PIL.Image as img
import matplotlib.pyplot as plt

path_image = r"D:\project_cnn\data\validate"

class Sample1:
    '''读取所有图像数据'''
    def read_data(self):
        self.img_arr=[]
        self.label = []
        #遍历系统文件夹里的文件名
        for name in os.listdir(path_image):
            data = name.split('.')
            x1 = int(data[1])/224
            y1 = int(data[2])/224
            x2 = int(data[3])/224
            y2 = int(data[4])/224
            c = int(data[5])
            self.label.append([x1,y1,x2,y2,c])
            imgs = img.open(r"{0}/{1}".format(path_image,name))
            # imgs.show()
            images=(np.array(imgs)/255-0.5)*2#/0.5
            self.img_arr.append(images)

        return self.img_arr,self.label

    def get_batch(self,set):
        '''获取随机数据采样的批次'''
        # self.read_data()
        self.get_data = []
        self.get_label = []
        for i in range(set):
            #生成图像个数长度内的一个随机数字
            num=np.random.randint(0,len(self.img_arr))
            # print(len(self.img_arr))
            # print(num)
            #将生成的随机数作为图像数据集的索引,把得到的随机图像数据累计起来
            self.get_data.append(self.img_arr[num])
            self.get_label.append(self.label[num])
            # imgs = (np.array(self.get_data) / 2 + 0.5) * 255
            # print(imgs.shape)
            #
            # imgs1 = img.fromarray(np.uint8(imgs[0]), 'RGB')
            # imgs1.show()
        return self.get_data,self.get_label

sample=Sample1()
# # sample.read_data()
# #查看随机图像批次形状
# print(np.shape(sample.get_batch(5)))
# print(np.shape(sample.get_batch(5)[0]))
# print(np.shape(sample.get_batch(5)[1]))
# #查看随机获得的图像数据
# print(sample.get_batch(1))
# xs,ys = sample.get_batch(5)
# imgs = (np.array(xs)/2+0.5)*255
# print(imgs.shape)
#
# imgs1 =img.fromarray(np.uint8(imgs[0]),'RGB')
# imgs1.show()
# print(xs)
# print(ys)
# print(np.shape(xs))
# print(np.shape(ys))