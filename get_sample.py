import os,shutil,random

def moveFile(fileDir,rate,tarDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    print(sample)
    for name in sample:
        shutil.move(fileDir +'/'+name, tarDir +'/'+ name)
    return

moveFile("D://project_cnn/negative_sample",0.6,"D://project_cnn/negative_train")
moveFile("D://project_cnn/negative_sample", 1, "D://project_cnn/negative_validate")
moveFile("D://project_cnn/positive_sample", 0.6, "D://project_cnn/positive_train")
moveFile("D://project_cnn/positive_sample", 1, "D://project_cnn/positive_validate")


