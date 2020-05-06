import os
import sys
import cv2
import pandas as pd
import numpy as np
# from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate

def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.

    INPUT
        directory: Folder to be created, called as "folder/".

    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)



def save_image(img,file_path,item):
    io.imsave(str(file_path + item), img)

def rotate_image(img,file_path,item,num):
    new_name=item.replace('.jpeg',' ')

    img1 = rotate(img, 90)
    new_item=new_name+'_'+num+'_'+'90.jpeg'
    save_image(img1,file_path,new_item)


    img2 = rotate(img, 120)
    new_item=new_name+'_'+num+'_'+'120.jpeg'
    save_image(img2,file_path,new_item)

    img3 = rotate(img, 180)
    new_item=new_name+'_'+num+'_'+'180.jpeg'
    save_image(img3,file_path,new_item)



    img4 = rotate(img, 270)
    new_item=new_name+'_'+num+'_'+'270.jpeg'
    save_image(img4,file_path,new_item)


    img5 = rotate(img, 360)
    new_item=new_name+'_'+num+'_'+'0.jpeg'
    save_image(img5,file_path,new_item)



    
def mirror_real_images(img,file_path,item):
    num='0'
    new_name=item.replace('.jpeg',' ')
    new_item=new_name+'_'+num+'.jpeg'
    save_image(img,file_path,new_item)
    img = cv2.flip(img, 1)
    num='1'
    new_item=new_name+'_'+num+'.jpeg'
    save_image(img,file_path,new_item)



def mirror_images(img,file_path,item):
    num='0'
    rotate_image(img,file_path,item,num)
    img = cv2.flip(img, 1)
    num='1'
    rotate_image(img,file_path,item,num)

def resize_real(item,img_path,file_path,cropx,cropy,img_size=256):
    img = io.imread(img_path+item)
    y,x,channel = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    img = img[starty:starty+cropy,startx:startx+cropx]
    img = resize(img, (256,256))
    if(np.mean(np.array(img)) == 0):
        print(item,'=Black')
    else:
        mirror_real_images(img,file_path,item)
        #io.imsave(str(file_path + item), img)
        print("Saving: ", item)
        


def resize_real_rotate(item,img_path,file_path,cropx,cropy,img_size=256):
    img = io.imread(img_path+item)
    y,x,channel = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    img = img[starty:starty+cropy,startx:startx+cropx]
    img = resize(img, (256,256))
    if(np.mean(np.array(img)) == 0):
        print(item,'=Black')
    else:
        mirror_images(img,file_path,item)
        #io.imsave(str(file_path + item), img)
        print("Saving: ", item)

def sep_fun(path,new_path):
    create_directory(new_path)
    new_path_0=new_path+'/no_dr'
    new_path_1=new_path+'/mild_dr'
    new_path_2=new_path+'/moderate_dr'
    new_path_3=new_path+'/severe_dr'
    new_path_4=new_path+'/p_dr'
    create_directory(new_path_0)
    create_directory(new_path_1)
    create_directory(new_path_2)
    create_directory(new_path_3)
    create_directory(new_path_4)
    count_t=0
    count_0=0
    count_1=0
    count_2=0
    count_3=0
    count_4=0
    df=pd.read_csv('../labels/trainLabels.csv', delimiter=',',index_col ='image')
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    for item in dirs:
        if(item.__contains__('.jpeg')):
            new_item=item.replace('.jpeg','')
            exists=new_item in df.index 
            if(exists):
                v=df.loc[new_item]
                count_t=count_t+1
                print(count_t)
                if(v.level==4):
                    save_path=new_path_4+'/'
                    resize_real_rotate(item,path,save_path,cropx=1800, cropy=1800, img_size=256)
                    count_4=count_4+1
                if(v.level==3):
                    save_path=new_path_3+'/'
                    resize_real_rotate(item,path,save_path,cropx=1800, cropy=1800, img_size=256)
                    count_3=count_3+1
                if(v.level==2):
                    save_path=new_path_2+'/'
                    resize_real_rotate(item,path,save_path,cropx=1800, cropy=1800, img_size=256)
                    count_2=count_2+1
                if(v.level==1):
                    save_path=new_path_1+'/'
                    resize_real_rotate(item,path,save_path,cropx=1800, cropy=1800, img_size=256)
                    count_1=count_1+1
                if(v.level==0):
                    save_path=new_path_0+'/'
                    resize_real(item,path,save_path,cropx=1800, cropy=1800, img_size=256)
                    count_0=count_0+1
    print('total='+str(count_t))
    print('0='+str(count_0))
    print('1='+str(count_1))
    print('2='+str(count_2))
    print('3='+str(count_3))
    print('4='+str(count_4))
    print('next')

if __name__ == '__main__':
    sep_fun(path='../data/train/', new_path='../data/train_classes')
    
  
    
