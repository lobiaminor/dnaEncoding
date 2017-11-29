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

imgdir = "./img/"
symbols = ['0', '1', '+', '-']

txt_list = glob.glob(os.path.join(imgdir, "*.txt"))

for txt in txt_list:
    with open(txt, 'r') as myfile:
        data = str(myfile.read().replace('\n', ''))
        print(txt + '\n')
        for sym in symbols:
            print(str(sym) + " : " + str(data.count(sym)))