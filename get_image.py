#Show&GetImages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 元となる画像の読み込み

def get_images(path):
    img = Image.open(path)
    img = img.resize((50,50))
    width,height = img.size
    img_pixels = np.array([[img.getpixel((i,j)) for j in range(height)] for i in range(width)])[:,:,0]
    img_pixels += np.array([[img.getpixel((i,j)) for j in range(height)] for i in range(width)])[:,:,1]
    img_pixels += np.array([[img.getpixel((i,j)) for j in range(height)] for i in range(width)])[:,:,2]
    #画像表示する場合
    img_pixels = np.sign(img_pixels/(255*3) * 2 -1)
    #plt.imshow(img_pixels.T, cmap='Oranges')
    #plt.show()
    #print(img_pixels)
    return img_pixels

#get_images(path)

if __name__ == '__main__':
    path = 'data/Pepper.bmp'
    path = 'data/Lenna.bmp'
    get_images(path)
