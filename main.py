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
import matplotlib.cm as cm
from skimage import color

def main():
    # Find all .jpg images in the img_original dir
    imgdir = "./img/"
    imagelist = glob.glob(os.path.join(imgdir, "*.jpg"))

    # Number of decomposition levels
    n = 3

    for filename in imagelist:
        #image = color.rgb2gray(img.imread(filename))
        #image = image.copy()
        image = readData(filename)

        #transformed = db.fwt97_2d(np.array(image, dtype=np.int64), n)
        transformed = wv.iwtn(image, n)
        scanned = scanning(get_subbands(transformed, n))

        sym = {"0":"0", "1":"1", "+":"+", "-":"-"} 
        sr_enc = sr_encoder.StackRunEncoder(sym)
        sr_dec = sr_decoder.StackRunDecoder(sym)

        encoded, runs, stacks = sr_enc.encode(scanned)  
        decoded = sr_dec.decode(encoded)

        decoded = reconstruct_subbands(unscanning(decoded, n))

        # Saving to file
        # name = filename.split("/")[-1].split(".")[0]
        # name = "results/depthmaps/" + name + "_encoded.txt"
        # with open(name,'w') as f:
        #     for s in encoded:
        #         f.write(str(s))

        result = wv.iiwtn(decoded, n)
        #result = db.iwt97_2d(decoded, n)

        # Calculate and print qbpp (qbits/px)
        qbpp = len(encoded)/(image.shape[0]*image.shape[1])

        # Measure entropy
        print(filename)
        #bpp = int(os.stat(filename).st_size)*8/(image.shape[0]*image.shape[1])
        #print("bpp = {}".format(bpp))
        print("qbits/px = {}".format(qbpp))
        print("OG entropy = {}".format(entropy_single(image)))
        print("Encoded entropy = {}".format(entropy_single(np.asarray(encoded), base=2)))
        print("Entropy = {} Shannon/symbol".format(entropy(runs, stacks)))

        # Show the image
        # plt.imshow(transformed)
        # plt.gray()
        # plt.show()


def readData(filename):
    array = np.loadtxt(filename, np.int64)
    #im = Image.fromarray(array, "I")
    return array


def entropy(runs, stacks, base=4):
    '''Calculates the entropy of an encoded image, represented by two dictionaries 
    containing the counts of the different sized stacks and runs respectively.

    Params:
        runs: dictionary where the keys represent run lengths and the values are their observed frequencies
        stacks: dictionary where keys are stack sizes and values their observed frequencies'''

    # To normalize the frequencies, first we need to obtain the total sum of the counts
    total = float(sum(runs.values()) + sum(stacks.values()))

    # Calculate the entropy using the probabilities
    entropy = 0
    for count in (list(runs.values()) + list(stacks.values())):
        if count != 0:
            norm = count/total
            entropy -= norm * np.math.log(norm, base)

    return entropy


def get_symbol2freq(vals):
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


def entropy_single(image, base=4):
    '''Calculates the entropy of the image passed as parameter. 
    
    Params:
        image:
        base: base of the logarithm used for the calculations. Default is 4.'''
    
    # Get a dictionary with the relative frequencies of each of the symbols
    # in image
    hist = get_symbol2freq(image.flatten()) 

    # Normalize the frequencies
    total = float(sum(hist.values()))

    # Calculate the entropy
    entropy = 0
    for count in hist.values():
        if count != 0:
            norm = count/total
            entropy -= norm * np.math.log(norm, base)

    return entropy


def calc_MSE(original, quantized):
    '''Get the Mean Squared Error for a given image and the corresponding reference.
    
    Params:
        original: reference, uncompressed image
        quantized: image being tested'''

    return (np.square(original - quantized)).mean(axis=None)


def get_subbands(image, n):
    """ Given a matrix representing the n-levels decomposition of an image,
    returns a list of its subbands. The ordering is as follows:
    -----------------
    |-1 | 0 |       | (Starting from the LL, subbands are ordered by
    |--------   3   |  level, clockwise)
    | 2 | 1 |       |
    |---------------|
    |       |       | 
    |   5   |   4   |
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
        subbands.append(image[0:start,   start:end])

    return subbands


def reconstruct_subbands(subbands):
    """Given the array of subbands, reconstructs the original matrix.
    The ordering of the subbands is assumed to be the same one described in the
    get_subbands method.

    Square images only!
    """
    # Last subband is half the size of the original image
    side = len(subbands[-1])*2
    image = np.zeros(shape=(side,side))

    # There are three subbands per decomposition level
    n = (len(subbands)-1)//3 

    # Add the LL
    end = side//pow(2, n)

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
    for i, band in enumerate(subbands[1:]):
        # Upper right
        if   i % 3 == 0:
            result = np.concatenate((result, band.T.flatten()))
        # Diagonal
        elif i % 3 == 1:
            result = np.concatenate((result, band.T.flatten()))
        # Lower left
        elif i % 3 == 2:
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
        for j in range(3):
            band_array = array[last:last+pow(band_side,2)]
            last = last+pow(band_side,2)
            # Upper right
            if   j % 3 == 0: 
                band = np.reshape(band_array, (band_side, band_side)).T
            # Diagonal
            elif j % 3 == 1:
                band = np.reshape(band_array, (band_side, band_side)).T
            # Lower left
            elif j % 3 == 2:
                band = np.reshape(band_array, (band_side, band_side)).T
            # Append it to the array of subbands
            subbands.append(band)

    return subbands


def quantize(vector, delta, minv=0, maxv=256):
    ''' Quantizes a vector using a given quantization step
    Params:
        vector: vector to be quantized
        delta: quantization step'''
    
    bins = np.linspace(minv, maxv, (abs(maxv - minv)/delta) + 1)
    indexes = np.digitize(vector, bins)

    return indexes

def dequantize(indexes, delta, minv):
    ''' Dequantizes a vector given the quantization step
    Params:
        indexes: vector of indexes to be dequantized
        delta: quantization step
        minv: minimum value of the original vector'''
    result = []
    for i in indexes:
        result.append(i*delta - delta/2.0 + minv)

    return result

def quantize_image(matrix, delta):
    ''' Quantize the image passed as parameter (it has to be a matrix)
    Params:
        matrix: numpy matrix representing the image
        delta: quantization step'''
    
    return np.apply_along_axis(quantize, 1, matrix, delta, minv=matrix.min(), maxv=matrix.max())


if __name__ == "__main__":
    main()