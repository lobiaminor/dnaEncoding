# dnaEncoding
pfe project 2017/2018 unice

# Questions

1. We encode 2D images as if they were long 1D sequences, so: how to decode? 
Our idea right now is to send a small header -two stacks- indicating the dimensions of the image. Then, the decoder can reconstruct the image because it knows its shape.

2. Also: how to scan through the image (line by line/col by col/diagonal maybe?). How to make this decision?? Based on what?
What we think today: the optimal thing is to have runs be as long as possible (having stacks grouped together). So possible options: use different scanning directions for each subband (horizontal/vertical/diagonal) see image
https://upload.wikimedia.org/wikipedia/commons/e/e0/Jpeg2000_2-level_wavelet_transform-lichtenstein.png



# To-do list
- Haar wavelet integer transform (aka Discrete wavelet transform?)
http://ieeexplore.ieee.org/abstract/document/586035/?reload=true
https://www.ece.uvic.ca/~frodo/publications/phdthesis.pdf

https://pywavelets.readthedocs.io/en/latest/
https://pywavelets.readthedocs.io/en/v0.3.0/ref/dwt-discrete-wavelet-transform.html

VERY GOOD: FINALLY NOT VERY GOOD
http://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html?highlight=haar
WHY is discrete not int -> int?

GOOD:
https://stackoverflow.com/a/15868889/5609680

Jose
- Wrap the decoder in a cool class. DONE


Both
- Start with the report (it will be awesome)