import PIL.Image as image
import numpy as np
import os

path1 =r'D:\project-cnn-get-backgrand\backgrand'

j=0
i=0
while i<1000:
    for name in os.listdir(path1):
        img_b = image.open(r'{0}/{1}'.format(path1,name))
        img_b = img_b.convert("RGB")
        img_b1 =img_b.resize((224,224))
        if os.path.isdir("D://project_cnn/negative_sample") != True:
            os.makedirs("D://project_cnn/negative_sample")
        img_b1.save("D://project_cnn/negative_sample/{num}.{x1}.{y1}.{x2}.{y2}.{c}.png".format(num=j, x1=0, y1=0, x2=0, y2=0,c=0))
        num1 = np.random.randint(1,21)
        img_y = image.open(r'D:\project-cnn-get-backgrand\yellow\{}.png'.format(num1))
        img_y1 = img_y.rotate(np.random.randint(-45,45))
        img_y2 = img_y1.resize((np.random.randint(50,100),np.random.randint(50,100)))
        w,h = img_y2.size
        r,g,b,a = img_y2.split()
        x = np.random.randint(0, 224 - w)
        y = np.random.randint(0,224-h)
        img_b1.paste(img_y2,(x,y),mask =a )
        # img_b1.show()
        if os.path.isdir("D://project_cnn/positive_sample") != True:
            os.makedirs("D://project_cnn/positive_sample")
        img_b1.save("D://project_cnn/positive_sample/{num}.{x1}.{y1}.{x2}.{y2}.{c}.png".format(num=j,x1=x,y1=y,x2=x+w,y2=y+h,c=1))
        j+=1
        i+=1
