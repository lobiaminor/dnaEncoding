# dnaEncoding
pfe project 2017/2018 unice

# Questions

1. We encode 2D images as if they were long 1D sequences, so: how to decode? 
Our idea right now is to send a small header -two stacks- indicating the dimensions of the image. Then, the decoder can reconstruct the image because it knows its shape.

# To-do list
- Haar wavelet integer transform (aka Discrete wavelet transform?)
http://ieeexplore.ieee.org/abstract/document/586035/?reload=true
https://www.ece.uvic.ca/~frodo/publications/phdthesis.pdf

https://pywavelets.readthedocs.io/en/latest/
https://pywavelets.readthedocs.io/en/v0.3.0/ref/dwt-discrete-wavelet-transform.html

VERY GOOD:
http://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html?highlight=haar

Jose
- Wrap the decoder in a cool class

Both
- Start with the report (it will be awesome)