import numpy as np
import os
import PIL.Image as img

path_image = r"D:\project_cnn\data\train"

class Sample:
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
        return self.get_data,self.get_label

# sample=Sample()
# # sample.read_data()
# #查看随机图像批次形状
# print(np.shape(sample.get_batch(5)))
# print(np.shape(sample.get_batch(5)[0]))
# print(np.shape(sample.get_batch(5)[1]))
# #查看随机获得的图像数据
# print(sample.get_batch(10))
# xs,ys = sample.get_batch(5)
# print(np.shape(xs))
# print(np.shape(ys))

# def net():
#     x=[10,224,224,3]
#     y=[10,5]
#
#     output=[10,5]
#     label_coord = y[:,:4]#[10,4]
#     label_conf = y[:,4:]#[10,1]
#     coord = output[:,:4]#[10,4]
#     conf = output[:,4:]#[10,1]
#     conf_out = sigmoid(conf)
#
#     coord_loss = (label_coord-coord)**2
#     conf_loss = (label_conf,conf)



