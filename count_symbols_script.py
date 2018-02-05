import os
import sys
import glob
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import wavelets as wv
from PIL import Image

imgdir = "./img/"
symbols = ['0', '1', '+', '-']

txt_list = glob.glob(os.path.join(imgdir, "*.txt"))

results = {'0':0, '1':0, '+':0, '-':0}

for txt in txt_list:
    with open(txt, 'r') as myfile:
        data = str(myfile.read().replace('\n', ''))
        print(txt + '\n')
        for sym in symbols:
            count = data.count(sym)
            print(str(sym) + " : " + str(count))
            results[sym] = results[sym] + count

a = {k: v / len(txt_list) for k, v in results.items()}
total = sum(a.values())
print("Total: {}".format(total))
b = {k: v / total for k, v in a.items()}
print(b)
print(a)