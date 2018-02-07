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
import configparser
from skimage import color

def main():
    # Parsing input arguments from settings.ini
    # For detailed description of the parameter, please refer to this settings.ini
    Config = configparser.ConfigParser()
    Config.read("settings.ini")

    # Input parameters
    imgdir = Config.get("input", "imgdir")
    extension = Config.get("input", "extension")
    n = int(Config.get("input", "n"))
    mode = Config.get("input", "mode")

    # Output parameters
    save_results = bool(int(Config.get("output", "save_results")))
    output_folder = Config.get("output", "output_folder")
    suffix = Config.get("output", "suffix")

    # Blob input images in a single list for easy recursion
    imagelist = glob.glob(os.path.join(imgdir, "*." + extension))

    for filename in imagelist:
        # Read the image
        image = color.rgb2gray(img.imread(filename))
        #image = image.copy()
        #image = readData(filename)

        # Initialize the Stack-Run encoder and decoder
        sym = {"0":"0", "1":"1", "+":"+", "-":"-"} 
        sr_enc = sr_encoder.StackRunEncoder(sym)
        sr_dec = sr_decoder.StackRunDecoder(sym)
        
        if mode == "lossless":
            # First, apply the selected wavelet transform to the image
            transformed97 = db.fwt97_2d(np.array(image, dtype=np.int64), n)
            transformed = wv.iwtn(image, n)

            # Next, scan the transformed image to convert it to a 1D signal 
            scanned = scanning(get_subbands(transformed, n))

            # Apply the stack-run coding algorithm
            encoded, runs, stacks = sr_enc.encode(scanned)  
            
            # Decode the image and reconstruct it so it becomes a 2D matrix again
            decoded = sr_dec.decode(encoded)
            decoded = reconstruct_subbands(unscanning(decoded, n))

            # Apply the inverse of the previous wavelet transform to obtain the decompressed img
            result = wv.iiwtn(decoded, n)
            #result = db.iwt97_2d(decoded, n)
        elif mode == "quantize":
            # Apply the wavelet transform to the image (this time it's not 
            # integer to integer, but we'll quantize afterwards)
            transformed = pywt.wavedec2(image, 'bior2.2', level=n)

            # Quantize each of the subbands with the appropriate quantization step 
            quantized = transformed

            # Next, scan the quantized image to convert it to a 1D signal 
            scanned = scanning(get_subbands(quantized, n))

            # Apply the stack-run coding algorithm
            encoded, runs, stacks = sr_enc.encode(scanned)  
            
            # Decode the image and undo the scanning to obtain the transformed version again
            decoded = sr_dec.decode(encoded)
            decoded = unscanning(decoded, n)

            # Dequantize
            dequantized = decoded

            # Apply the inverse of the previous wavelet transform to obtain the decompressed img
            result = wv.iiwtn(dequantized, n)
        else:
            print("Incorrect 'mode' parameter, please input one of the possible options.")
            break

        # Saving the encoded string to a file
        if save_results:
            name = os.path.basename(filename).split(".")[0] + suffix
            name = os.path.join(output_folder, name) 
            with open(name,'w') as f:
                for s in encoded:
                    f.write(str(s))

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

        print("Haar_entropy = {}".format(entropy_by_subbands(transformed)))
        print("CDF97_entropy = {}".format(entropy_by_subbands(transformed97)))

        # Show the image
        # plt.imshow(result)
        # plt.gray()
        # plt.show()


def readData(filename):
    array = np.loadtxt(filename, np.int64)
    #im = Image.fromarray(array, "I")
    return array


def entropy_by_subbands(subbands, base=2):
    n = len(subbands) - 1
    entropy = entropy_single(subbands[0], base=base) / pow(4, n)
    
    for i, band in enumerate(subbands[1:]):
        entropy += entropy_single(band, base=base) / pow(4, n-i)

    return entropy


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


def entropy_single(image, base=4):
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


def mse(original, decompressed):
    '''Get the Mean Squared Error for a given image and the corresponding reference.
    
    Params:
        original: reference, uncompressed image
        decompressed: image being tested'''

    return (np.square(original - decompressed)).mean(axis=None)


def get_subbands(image, n):
    """ Given a matrix representing the n-levels decomposition of an image,
    returns a list of its subbands. The ordering is as follows:
    -----------------
    | 0 | 1 |       | Starting from the LL, subbands are ordered by
    |--------   4   | level. For the following part, the naming is as follows:  
    | 2 | 3 |       |   0: cA2
    |---------------|   1: cH2 
    |       |       |   2: cV2
    |   5   |   6   |   3: cD2
    |       |       |   4: cH1 ...
    -----------------

    Be careful, the returned subbands are views of the original matrix. That
    means they will be modified if the original image changes.

    Params:
    - image: numpy matrix representing the image (square matrix)
    - n: number of decomposition levels

    Returns:
    A list [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)] 

    Square images only!
    """
    # To make it work for non-square images, take width/height instead of
    # only side
    side = image.shape[0]
    
    # LL
    subbands = []
    end = side//pow(2, n)
    subbands.append(image[0:end,0:end])

    # For each decomposition level, extract the subbands
    for i in reversed(range(n)):
        start = side//pow(2, i+1)
        end   = 2*start
        
        h = image[start:end, 0:start]
        v = image[0:start,   start:end]
        d = image[start:end, start:end]

        subbands.append((h, v, d))

    return subbands


def reconstruct_subbands(subbands):
    """Given the array of subbands, reconstructs the original matrix.
    The ordering of the subbands is assumed to be the same one described in the
    get_subbands method.

    Square images only!
    """
    # Last subband is half the size of the original image
    side = len(subbands[-1][0])*2
    image = np.zeros(shape=(side,side))

    # There are len(subbands)-1 decomposition levels, because the LL is separated
    n = len(subbands)-1 

    # Add the LL
    end = side//pow(2, n)
    image[0:end, 0:end] = subbands[0]

    for i in reversed(range(n)):
        start = side//pow(2, i+1)
        end   = 2*start
        idx = n - i

        image[start:end, 0:start]  = subbands[idx][0]
        image[0:start, start:end]  = subbands[idx][1]
        image[start:end, start:end]= subbands[idx][2]

    return image


def scanning(subbands):
    """ Given the array of subbands, scans through each one of them in 
    the corresponding direction and returns a 1D array.

    Params:
        subbands: list containing the subbands, in the following way
                  [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)] (see get_subbands)
    """
    # 1. Scan the LL
    result = subbands[0].flatten()
    
    # 2. Scan the rest of subbands
    for level_bands in subbands[1:]:
        # Upper right (horizontal detail)
        result = np.concatenate((result, level_bands[0].T.flatten()))
        # Lower left (vertical detail)
        result = np.concatenate((result, level_bands[1].T.flatten()))
        # Diagonal
        result = np.concatenate((result, level_bands[2].T.flatten()))

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
        level_bands = [] # Used to store the 3 subbands belonging to a level
        for j in range(3):
            band_array = array[last:last+pow(band_side,2)]
            last = last+pow(band_side,2) # After extracting each subband, we update this pointer
            # Upper right
            if   j % 3 == 0: 
                level_bands.append(np.reshape(band_array, (band_side, band_side)).T)
            # Lower left
            elif j % 3 == 1:
                level_bands.append(np.reshape(band_array, (band_side, band_side)).T)
            # Diagonal
            elif j % 3 == 2:
                level_bands.append(np.reshape(band_array, (band_side, band_side)).T)
        
        # Append a tuple composed of the level's band to the array of subbands
        subbands.append((level_bands[0], level_bands[1], level_bands[2]))

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