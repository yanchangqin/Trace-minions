import numpy as np
import os
import PIL.Image as img


path_image = r"D:\project_cnn\data\validate"

class Sample1:
    '''读取所有图像数据'''
    def read_data(self):
        self.img_dir=[]
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
            self.dir = os.path.abspath(r'{0}/{1}'.format(path_image,name))
            self.img_dir.append(self.dir)

        return self.img_dir,self.label

    def get_batch(self,set):
        '''获取随机数据采样的批次'''
        # self.read_data()
        self.get_data = []
        self.get_label = []
        for i in range(set):
            #生成图像个数长度内的一个随机数字
            num=np.random.randint(0,len(self.img_dir))
            imgs = img.open(self.img_dir[num])
            images = (np.array(imgs) / 255 - 0.5) * 2  # /0.5

            self.get_data.append(images)
            # print(self.get_data)
            self.get_label.append(self.label[num])

        return self.get_data,self.get_label
