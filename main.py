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
    # img_src = "img_original/{}.jpg".format(sys.argv[1])
    # dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, img_src)
    
    # Find all .jpg images in the img_original dir
    imgdir = "./img/"
    imagelist = glob.glob(os.path.join(imgdir, "*.jpg"))

    for filename in imagelist:
        image = img.imread(filename)

        transformed = wv.iwtn(image, 2)
        
        # name = filename.split("/")[-1].split(".")[0]
        # name = name + "_encoded.txt"

        sym = {"0":"0", "1":"1", "+":"+", "-":"-"} 
        sr_enc = sr_encoder.StackRunEncoder(sym)
        sr_dec = sr_decoder.StackRunDecoder(sym)

        encoded = sr_enc.encode(transformed.flatten())  
        decoded = sr_dec.decode(encoded)

        decoded = np.reshape(decoded, transformed.shape)

        # with open(name,'w') as f:
        #     for s in encoded:
        #         f.write(str(s))

        result = wv.iiwtn(decoded, 2)

        # Calculate and print qbpp (qbits/px)
        qbpp = len(encoded)/(image.shape[0]*image.shape[1])
        
        # Measure entropy
        print(filename)
        print("qbits/px = {}".format(qbpp))
        print("Entropy = {} nats/symbol".format(entropy(encoded)))

        # Show the image
        # plt.imshow(result)
        # plt.gray()
        # plt.show()

def get_symbol2freq(vals):
    """ Creates a dictionary where each symbol has its frequency associated to it"""
    hist = {}

    # Get the histogram
    for v in vals:
        if v in hist:
            hist[v] = hist[v] + 1
        else:
            hist[v] = 1

    return hist

def entropy(signal):
    """ Calculate the entropy of the image passed as parameter (matrix)"""
    hist = get_symbol2freq(signal)

    # Normalize the freqs
    total = float(sum(hist.values()))

    entropy = 0
    for count in hist.values():
        if count != 0:
            norm = count/total
            entropy += norm * np.math.log(norm, 4)

    return (entropy*(-1))

if __name__ == "__main__":
    main()

# By rows
# a.flatten()
# By cols
# a.T.flatten()
# To reconstruct
# np.reshape(a, (x,y)) -> with a -1 on the dimension we dont want to enforce
# TODO: Scan through the img_original and encode it

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