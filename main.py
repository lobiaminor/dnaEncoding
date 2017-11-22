import os
import sys
import sr_decoder
import sr_encoder
# import pywt
import glob
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import wavelets as wv
from PIL import Image

def main():
    # Read images from the image dir
    # img_src = "img/{}.jpg".format(sys.argv[1])
    # dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, img_src)
    
    # Find all .jpg images in the img dir    
    imgdir = "./img/"
    imagelist = glob.glob(os.path.join(imgdir, "*.jpg"))

    for filename in imagelist:
        image = img.imread(filename)

        transformed = wv.iwtn(image, 2)
        
        sym = {"0":"A", "1":"T", "+":"C", "-":"G"} 
        sr_enc = sr_encoder.StackRunEncoder(sym)
        sr_dec = sr_decoder.StackRunDecoder(sym)

        encoded = sr_enc.encode(transformed.flatten())  
        decoded = sr_dec.decode(encoded)

        decoded = np.reshape(decoded, transformed.shape)

        result = wv.iiwtn(decoded, 2)

        # Calculate and print qbpp (qbits/px)
        qbpp = len(encoded)/(image.shape[0]*image.shape[1])
        print("{}: {}".format(filename, qbpp))
        # Show the image
        # plt.imshow(result)
        # plt.gray()
        # plt.show()



if __name__ == "__main__":
    main()

# By rows
# a.flatten()
# By cols
# a.T.flatten()
# To reconstruct
# np.reshape(a, (x,y)) -> with a -1 on the dimension we dont want to enforce
# TODO: Scan through the img and encode it

# # Testing stuff
# sym = {"0":"A", "1":"T", "+":"C", "-":"G"} 
# sr_enc = sr_encoder.StackRunEncoder(sym)
# sr_dec = sr_decoder.StackRunDecoder(sym)

# s = "0,0,0,0,0,1,0,0,0,25,0,0".split(",")
# s = [int(x) for x in s]
# s1 = s[:-2]
# print(s)
# print(s1)
# a = sr_enc.encode(s)
# a1 = sr_enc.encode(s1)
# print(a)
# print(a1)
# b = sr_dec.decode(a)
# print(b)