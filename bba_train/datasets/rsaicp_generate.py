import shutil
import os
from tqdm import tqdm

        
if __name__ == '__main__':
    os.makedirs('RSAICP_MS37')
    os.makedirs('RSAICP_MS37/images')
    os.makedirs('RSAICP_MS37/labelTxt')
    
    train_images = './rsaicp_ms37/train/images'
    train_labels = './rsaicp_ms37/train/labelTxt'
    temp1 = [os.path.join(train_images, img) for img in os.listdir(train_images)]
    temp2 = [os.path.join(train_labels, lb) for lb in os.listdir(train_labels)]
    for img in tqdm(temp1):
      shutil.copy(img,'RSAICP_MS37/images')
    for lb in tqdm(temp2):
      shutil.copy(lb,'RSAICP_MS37/labelTxt')
    with open('RSAICP_MS37/train.txt', 'w+') as ftxt:
      for img in tqdm(os.listdir(train_images)):
        ftxt.writelines(os.path.splitext(img)[0] + '\n')
    '''
    val_images = './rsaicp_608/val/images'
    val_labels = './rsaicp_608/val/labelTxt'
    temp3 = [os.path.join(val_images, img) for img in os.listdir(val_images)]
    temp4 = [os.path.join(val_labels, lb) for lb in os.listdir(val_labels)]
    for img in temp3:
      shutil.copy(img,'RSAICP_608/images')
    for lb in temp4:
      shutil.copy(lb,'RSAICP_608/labelTxt')
    with open('RSAICP_608/val.txt', 'w+') as ftxt:
      for img in os.listdir(val_images):
        ftxt.writelines(os.path.splitext(img)[0] + '\n')
    '''

