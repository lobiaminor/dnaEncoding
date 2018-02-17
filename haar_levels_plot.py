import os
import sys
import stackrun as sr
# import pywt
import nineseven as db
import glob
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import haar as wv
import main as m
from PIL import Image

def main():
    # Read images from the image dir
    # img_src = "img_original/{}.jpg".format(sys.argv[1])
    # dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, img_src)
    
    # Find all .jpg images in the img_original dir
    imgdir = "./img/"
    imagelist = glob.glob(os.path.join(imgdir, "*.jpg"))

    entropies = np.zeros(8)
    imagelist = imagelist[1:4] # In case we don't want to use all of the images

    for filename in imagelist:
        image = img.imread(filename)
        image = image.copy()
        
        width = image.shape[0]
        height= image.shape[1]

        for n in range(1,9):
            print(".")
            #transformed = db.fwt97_2d(np.array(image, dtype=np.int64), n)
            transformed = wv.iwtn(image, n)

            sym = {"0":"0", "1":"1", "+":"+", "-":"-"} 
            sr_enc = sr.StackRunEncoder(sym)
            sr_dec = sr.StackRunDecoder(sym)
            
            # Next, scan the transformed image to convert it to a 1D signal 
            scanned = m.scanning(m.get_subbands(transformed, n))

            # # Apply the stack-run coding algorithm
            encoded = sr_enc.encode(scanned)  
            # # Decode the image and reconstruct it so it becomes a 2D matrix again
            # decoded = sr_dec.decode(encoded)
            # decoded = m.reconstruct_subbands(m.unscanning(decoded, n, width, height), width, height)
            # # Apply the inverse of the previous wavelet transform to obtain the decompressed img
            # result = wv.iiwtn(decoded, n)

            entropies[n-1] = entropies[n-1] + (len(encoded) * m.entropy_by_subbands(m.get_subbands(transformed, n), base=2) / (width*height))

    # Measure entropy
    entropies = entropies/len(imagelist)
    print(entropies)

    plt.plot(range(1,9), entropies)
    plt.ylabel("Entropy (Shannon/pixel)")
    plt.xlabel("Number of decomposition levels")
    plt.show()


def histogram(vals):
    '''Given an array of symbols, returns a dictionary where the keys are those symbols and
    the values are their counts.
    
    Params:
        vals: array to be counted'''

    # Get the histogram
    hist = {}
    for v in vals:
        if v in hist:
            hist[v] = hist[v] + 1
        else:
            hist[v] = 1

    return hist


def entropy_by_subbands(subbands, base=2):
    '''Calculates the entropy of the transformed image passed as parameter. 
    
    Params:
        subbands: array containing the coefficients of the subbands of the transformed image
                  (same format as get_subbands is expected)
        base: base of the logarithm used for the calculations. Default is 2.'''
    n = len(subbands) - 1
    entropy = entropy_single(subbands[0], base=base) / pow(4, n)
    
    for i, bands in enumerate(subbands[1:]):
        for band in bands:
            entropy += entropy_single(band, base=base)/ pow(4, n-i)

    return entropy


def entropy(image, base=4):
    '''Calculates the entropy of the image passed as parameter. 
    
    Params:
        image:
        base: base of the logarithm used for the calculations. Default is 4.'''
    
    # Get a dictionary with the relative frequencies of each of the symbols
    # in image
    hist = histogram(image.flatten()) 

    # Normalize the frequencies
    total = float(sum(hist.values()))

    # Calculate the entropy
    entropy = 0
    for count in hist.values():
        if count != 0:
            norm = count/total
            entropy -= norm * np.math.log(norm, base)

    return entropy


if __name__ == "__main__":
    main()
