import shutil
import os

        
if __name__ == '__main__':
    os.makedirs('RSAICP')
    os.makedirs('RSAICP/images')
    os.makedirs('RSAICP/labelTxt')
    
    train_images = './rsaicp/train/images'
    train_labels = './rsaicp/train/labelTxt'
    temp1 = [os.path.join(train_images, img) for img in os.listdir(train_images)]
    temp2 = [os.path.join(train_labels, lb) for lb in os.listdir(train_labels)]
    for img in temp1:
      shutil.copy(img,'RSAICP/images')
    for lb in temp2:
      shutil.copy(lb,'RSAICP/labelTxt')
    with open('RSAICP/train.txt', 'w+') as ftxt:
      for img in os.listdir(train_images):
        ftxt.writelines(os.path.splitext(img)[0] + '\n')
    
    val_images = './rsaicp/val/images'
    val_labels = './rsaicp/val/labelTxt'
    temp3 = [os.path.join(val_images, img) for img in os.listdir(val_images)]
    temp4 = [os.path.join(val_labels, lb) for lb in os.listdir(val_labels)]
    for img in temp3:
      shutil.copy(img,'RSAICP/images')
    for lb in temp4:
      shutil.copy(lb,'RSAICP/labelTxt')
    with open('RSAICP/val.txt', 'w+') as ftxt:
      for img in os.listdir(val_images):
        ftxt.writelines(os.path.splitext(img)[0] + '\n')

