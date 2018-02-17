import glymur
import os
import sys
import math
import stackrun as sr
import pywt
import nineseven as db
import glob
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import haar as wv
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

        # # Check if the image has appropriate dimensions
        # # (square and sides are powers of 2)
        width = image.shape[0]
        height= image.shape[1]
        
        # # If it's rectangular, fill with zeros to make it square
        # if(width != height):
        #     diff = width - height
        #     p = abs(diff)/2 # The amount of zeroes we need to add in each side
        #     # Diff > 0, we need to pad the top and the bottom
        #     if(diff > 0):
        #         # np.pad (mode 'constant') adds zeros: (top, bottom)(left, right)
        #         image = np.pad(image,((math.floor(p), math.ceil(p)),(0,0)), 'constant')
        #     elif(diff < 0):
        #         image = np.pad(image,((0,0),(math.floor(p), math.ceil(p))), 'constant')

        # Initialize the Stack-Run encoder and decoder
        sym = {"0":"0", "1":"1", "+":"+", "-":"-"} 
        sr_enc = sr.StackRunEncoder(sym)
        sr_dec = sr.StackRunDecoder(sym)
        
        if mode == "lossless":
            # First, apply the selected wavelet transform to the image
            # transformed = db.fwt97_2d(np.array(image, dtype=np.int64), n)
            transformed = wv.iwtn(image, n)

            # Next, scan the transformed image to convert it to a 1D signal 
            scanned = scanning(get_subbands(transformed, n))

            # Apply the stack-run coding algorithm
            encoded = sr_enc.encode(scanned)  
            
            # Decode the image and reconstruct it so it becomes a 2D matrix again
            decoded = sr_dec.decode(encoded)
            decoded = reconstruct_subbands(unscanning(decoded, n, width, height), width, height)

            # Apply the inverse of the previous wavelet transform to obtain the decompressed img
            result = wv.iiwtn(decoded, n)
            # result = db.iwt97_2d(decoded, n)
        elif mode == "quantize":
            # Apply the wavelet transform to the image 
            # This time it's not integer to integer, but we'll quantize afterwards
            # 'periodization' mode ensures that the coeffs have the minimum size, that is,
            # submultiples of the original dimensions
            transformed = pywt.wavedec2(image, 'bior2.2', level=n, mode='periodization')

            # Quantize each of the subbands with the appropriate quantization step 
            quantized, steps = quantize_subbands(transformed)

            # Next, scan the quantized image to convert it to a 1D signal 
            scanned = scanning(quantized)

            # Apply the stack-run coding algorithm
            encoded = sr_enc.encode(scanned)  
            
            # Decode the image and undo the scanning to obtain the transformed version again
            decoded = sr_dec.decode(encoded)
            decoded = unscanning(decoded, n, width, height)

            # Dequantize
            dequantized = dequantize_subbands(decoded, steps)

            # Apply the inverse of the previous wavelet transform to obtain the decompressed img
            result = pywt.waverec2(dequantized, 'bior2.2', mode='periodization')
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
        sympp = len(encoded)/(image.shape[0]*image.shape[1])

        # Measure entropy
        entropy_encoded = entropy(np.asarray(encoded), base=4)*len(encoded)/(width*height)
        
        print(filename)
        print("Symbols/px = {}".format(sympp))
        print("Original image entropy = {}".format(entropy(image, base=2)))
        if mode == "lossless":
            print("Transformed image entropy = {}".format(entropy_by_subbands(get_subbands(transformed, n))))
        elif mode == "quantize":
            # We are only interested in the entropy after quantization
            # print("Transformed image entropy = {}".format(entropy_by_subbands(transformed)))
            print("Quantized image entropy = {}".format(entropy_by_subbands(quantized)))
        print("Encoded entropy = {} Shannon/pixel".format(entropy_encoded))
        print("MSE = {}".format(mse(image, result)))
        
        # Show the image
        # plt.imshow(result)
        # plt.gray()
        # plt.show()


def readData(filename):
    array = np.loadtxt(filename, np.int64)
    #im = Image.fromarray(array, "I")
    return array


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


def entropy_by_subbands(subbands, base=2):
    '''Calculates the entropy of the transformed image passed as parameter. 
    
    Params:
        subbands: array containing the coefficients of the subbands of the transformed image
                  (same format as get_subbands is expected)
        base: base of the logarithm used for the calculations. Default is 2.'''
    n = len(subbands) - 1
    ent = entropy(subbands[0], base=base) / pow(4, n)
    
    for i, bands in enumerate(subbands[1:]):
        for band in bands:
            ent += entropy(band, base=base)/ pow(4, n-i)

    return ent


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


def reconstruct_subbands(subbands, width, height):
    """Given the array of subbands, reconstructs the original matrix.
    The ordering of the subbands is assumed to be the same one described in the
    get_subbands method.

    Params:
        subbands: list containing the subbands, in the following way
                  [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)] (see get_subbands)
        width, height: dimensions of the original image
    """

    image = np.zeros(shape=(width, height))

    # There are len(subbands)-1 decomposition levels, because the LL is separated
    n = len(subbands)-1 

    # Add the LL
    end_h = width//pow(2, n)
    end_v = height//pow(2, n)

    image[0:end_h, 0:end_v] = subbands[0]

    for i in reversed(range(n)):
        start_h = width//pow(2, i+1)
        end_h   = 2*start_h
        start_v = height//pow(2, i+1)
        end_v   = 2*start_v
        idx = n - i

        image[start_h:end_h, 0:start_v]    = subbands[idx][0]
        image[0:start_h, start_v:end_v]    = subbands[idx][1]
        image[start_h:end_h, start_v:end_v]= subbands[idx][2]

    return image


def scanning(subbands):
    """ Given the array of subbands, scans through each one of them in 
    the corresponding direction and returns a 1D array.

    Params:
    - subbands: list containing the subbands, in the following way
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


def unscanning(array, n, width, height):
    """ Given the 1D array representing the transformed image, reconstructs
    its original shape (undoes the scanning).
    
    Params:
    - array: numpy array (1D) representing the transformed image
    - n: number of decomposition levels
    - width, height: dimensions of the original image
    """

    # 1. Retrieve the LL
    width_ll = width // pow(2, n)
    height_ll= height// pow(2, n)
    
    last = width_ll*height_ll # Last element of the array that was analyzed 
    ll = np.reshape(array[0:last], (width_ll, height_ll))

    subbands = []
    subbands.append(ll)

    # 2. Retrieve the rest of the subbands
    for i in reversed(range(n)):
        band_w = width//pow(2, i+1)
        band_h = height//pow(2, i+1)
        band_pixels = band_w*band_h # Number of px in each subband of this level
        level_bands = [] # Used to store the 3 subbands belonging to a level
        for j in range(3):
            band_array = array[last:last+band_pixels]
            last = last+band_pixels # After extracting each subband, we update this pointer
            # Upper right
            if   j % 3 == 0: 
                level_bands.append(np.reshape(band_array, (band_w, band_h)).T)
            # Lower left
            elif j % 3 == 1:
                level_bands.append(np.reshape(band_array, (band_w, band_h)).T)
            # Diagonal
            elif j % 3 == 2:
                level_bands.append(np.reshape(band_array, (band_w, band_h)).T)
        
        # Append a tuple composed of the level's band to the array of subbands
        subbands.append((level_bands[0], level_bands[1], level_bands[2]))

    return subbands


def quantize(vector, delta):
    ''' Quantizes a vector using a given quantization step
    Params:
    - vector: vector to be quantized
    - delta: quantization step'''

    return np.round(vector/delta)


def quantize_band(matrix, delta):
    ''' Quantize the image passed as parameter (it has to be a matrix)
    Params:
    - matrix: numpy matrix representing the image
    - delta: quantization step'''
    
    return np.apply_along_axis(quantize, 1, matrix, delta)


def quantize_subbands(subbands, steps=False):
    ''' Quantizes the transformed image passed as parameter (it has to be represented
    as a list of subbands, in the same way get_subbands returns it)
    
    Params:
    - subbands: list containing the subbands, in the following way
                [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)] (see get_subbands)
    - steps:(optional) list containing the quantization steps used for the subbands corresponding
            to each decomposition level. If not specified, the quantization step is automatically 
            chosen.
    '''
    
    quantized = []
    
    # If the quantization step is not defined, assign it    
    if not steps:
        steps = []
        n = len(subbands)
        for i in reversed(range(n)): 
            # Quant step is, for each level:
            # 256 / n+3, n+2, n, ..., 4 
            steps.append(256.0/pow(2, (i+4)))


    quantized.append(quantize_band(subbands[0], steps[0]))
    
    for i, bands in enumerate(subbands[1:]):
        quantized.append((
            quantize_band(bands[0], steps[i+1]),
            quantize_band(bands[1], steps[i+1]),
            quantize_band(bands[2], steps[i+1])))

    return quantized, steps


def dequantize(indexes, delta):
    ''' Dequantizes a vector given the quantization step
    Params:
        indexes: vector of indexes to be dequantized
        delta: quantization step'''
    return indexes.astype(np.int64)*delta

def dequantize_band(matrix, delta):
    ''' Dequantize the subband passed as parameter (it has to be a matrix)
    Params:
        matrix: numpy matrix representing the quantized subband
        delta: quantization step
        '''
    
    return np.apply_along_axis(dequantize, 1, matrix, delta)


def dequantize_subbands(subbands, deltas):
    dequantized = []
    dequantized.append(dequantize_band(subbands[0], deltas[0]))
    
    for i, bands in enumerate(subbands[1:]):
        dequantized.append((
            dequantize_band(bands[0], deltas[i+1]),
            dequantize_band(bands[1], deltas[i+1]),
            dequantize_band(bands[2], deltas[i+1])))
        
    return dequantized


if __name__ == "__main__":
    main()