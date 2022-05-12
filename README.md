# Temporal-evolution-inference-with-compact-feature-representation-for-talking-face-video-compression

This repository contains the source code for the paper “Beyond Keypoint Coding: Temporal Evolution Inference with Compact Feature Representation for Talking Face
Video Compression” by Bolin Chen, Zhao Wang, Binzhe Li, Rongqun Lin, Shiqi Wang, and Yan Ye

The DCC keynote video can be found in https://www.youtube.com/watch?v=7en3YYT1QfU

# Training

python run.py 
 
the gpu number and save dir can be modified in run.py

# Inference

python encode.py 
 
python decode.py

# Reference

The training code refers to the FOMM: https://github.com/AliaksandrSiarohin/first-order-model

The arithmetic-coding refers to https://github.com/nayuki/Reference-arithmetic-coding



# License


--------------------------- LICENSE FOR Reference arithmetic coding --------------------------------

Copyright © 2020 Project Nayuki. (MIT License)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

* The Software is provided "as is", without warranty of any kind, express or
  implied, including but not limited to the warranties of merchantability,
  fitness for a particular purpose and noninfringement. In no event shall the
  authors or copyright holders be liable for any claim, damages or other
  liability, whether in an action of contract, tort or otherwise, arising from,
  out of or in connection with the Software or the use or other dealings in the
  Software.


