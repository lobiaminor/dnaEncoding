import os
import sys
import sr_decoder
import sr_encoder
# import pywt
import nineseven as db
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
        image = image.copy()

        transformed = wv.iwtn(image, 3) #db.fwt97_2d(image, 3)
        
        # name = filename.split("/")[-1].split(".")[0]
        # name = name + "_encoded.txt"

        sym = {"0":"0", "1":"1", "+":"+", "-":"-"} 
        sr_enc = sr_encoder.StackRunEncoder(sym)
        sr_dec = sr_decoder.StackRunDecoder(sym)

        encoded, runs, stacks = sr_enc.encode(transformed.flatten())  
        decoded = sr_dec.decode(encoded)

        decoded = np.reshape(decoded, transformed.shape)

        # with open(name,'w') as f:
        #     for s in encoded:
        #         f.write(str(s))

        result = wv.iiwtn(decoded, 3) #db.iwt97_2d(decoded, 3)

        # Calculate and print qbpp (qbits/px)
        qbpp = len(encoded)/(image.shape[0]*image.shape[1])

        # Measure entropy
        print(filename)
        print("qbits/px = {}".format(qbpp))
        print("OG Entropy = {}".format(entropy_single(image)))
        print("Entropy = {} nats/symbol".format(entropy(runs, stacks)))

        # Show the image
        # plt.imshow(result)
        # plt.gray()
        # plt.show()


def entropy(runs, stacks):
    # Normalize the freqs
    total = float(sum(runs.values()) + sum(stacks.values()))

    entropy = 0
    
    for count in (list(runs.values()) + list(stacks.values())):
        if count != 0:
            norm = count/total
            entropy += norm * np.math.log(norm, 4)

    return -entropy


def get_symbol2freq(vals):
    hist = {}

    # Get the histogram
    for v in vals:
        if v in hist:
            hist[v] = hist[v] + 1
        else:
            hist[v] = 1

    return hist


def entropy_single(image):
    hist = get_symbol2freq(image.flatten()) 

    # Normalize the freqs
    total = float(sum(hist.values()))

    entropy = 0
    
    for count in hist.values():
        if count != 0:
            norm = count/total
            entropy += norm * np.math.log(norm, 4)

    return -entropy

if __name__ == "__main__":
    main()


def get_subbands(image, n):
    """ Given a matrix representing the n-levels decomposition of an image,
    returns a list of its subbands. The ordering is as follows:
    -----------------
    | 0 | 1 |       | (Starting from the LL, subbands are ordered by
    |--------   4   |  level, clockwise)
    | 3 | 2 |       |
    |---------------|
    |       |       | 
    |   6   |   5   |
    |       |       |
    -----------------

    Be careful, the returned subbands are views of the original matrix. That
    means they will be modified if the original image changes.

    Params:
    - image: numpy matrix representing the image (square matrix)
    - n: number of decomposition levels

    Square images only!
    """
    # To make it work for non-square images, take width/height instead of
    # only side
    side = image.shape[0]
    
    # LL
    subbands = []
    end = side//pow(2, n)
    subbands.append(image[0:end,0:end])

    # For each decomposition level, extract the subbands, clockwise
    for i in reversed(range(n)):
        start = side//pow(2, i+1)
        end   = 2*start
        
        subbands.append(image[start:end, 0:start]) 
        subbands.append(image[start:end, start:end])
        subbands.append(image[0:start, start:end])

    return subbands


def reconstruct_subbands(subbands):
    """Given the array of subbands, reconstructs the original matrix.
    The ordering of the subbands is assumed to be the same one described in the
    get_subbands method.

    Square images only!
    """
    # Last subband is half the size of the original image
    side = len(subbands[-1])*2 
    image = np.zeros(shape=(size,size))

    # There are three subbands per decomposition level
    n = (len(subbands)-1)//3 

    # Add the LL
    end = size//pow(2, n)
    image[0:end, 0:end] = subbands[0]

    for i in reversed(range(n)):
        start = side//pow(2, i+1)
        end   = 2*start
        idx = n - i - 1

        image[start:end, 0:start]  = subbands[3*idx + 1]
        image[start:end, start:end]= subbands[3*idx + 2]
        image[0:start, start:end]  = subbands[3*idx + 3]

    return image


def scanning(subbands):
    """ Given the array of subbands, scans through each one of them in 
    the corresponding direction and returns a 1D array.
    """
    # 1. Scan the LL
    result = subbands[0].flatten()
    
    # 2. Scan the rest of subbands
    for i, band in enumerate(subbands):
        # Upper right
        if i % 3 == 1:
            result = np.concatenate((result, band.flatten()))
        # Diagonal
        elif i % 3 == 2:
            result = np.concatenate((result, band.flatten()))
        # Lower left
        elif i % 3 == 0:
            result = np.concatenate((result, band.T.flatten()))

    return result


def unscanning(array, n):
    """ Given the 1D array representing the transformed image, reconstructs
    its original shape (undoes the scanning).
    
    Params:
    - array: numpy array (1D) representing the transformed image
    - n: number of decomposition levels

    Square images only!
    """

    # Original image's dimensions
    side = int(np.sqrt(len(array)))

    # 1. Retrieve the LL
    side_ll = side // pow(2, n)
    last = pow(side_ll,2) # Last element of the array that was analyzed 
    ll = np.reshape(array[0:last], (side_ll, side_ll))

    subbands = []
    subbands.append(ll)

    # 2. Retrieve the rest of the subbands
    for i in reversed(range(n)):
        band_side = side//pow(2, i+1)
        # Using [1,3] for consistency with the numbering in the scanning method
        for j in range(1,4):
            band_array = array[last:last+pow(band_side,2)]
            last = last+pow(band_side,2)
            # Upper right
            if j % 3 == 1: 
                band = np.reshape(band_array, (band_side, band_side))
            # Diagonal
            elif j % 3 == 2:
                band = np.reshape(band_array, (band_side, band_side))
            # Lower left
            elif j % 3 == 0:
                band = np.reshape(band_array, (band_side, band_side)).T
            # Append it to the array of subbands
            subbands.append(band)

    return subbands



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