include_libtorch_VS.md - this file contains instructions for connecting the LibTorch library in the Visual Studio development environment


LibTorch is a CPP API Pytorch

Pytorch - it is a neutral home for the deep learning (https://pytorch.org/foundation)

npy.hpp - contain program code, which allows you to read input data from the .npy file and convert it into a format suitable for processing on CPP (in my case it is a vector<double> and then just double*)



ConsoleApplication1.cpp - contain main method. this method contains the main functionality: 

1. reading a pre-trained model from a .pt file, 

2. reading input data from a .npy file, 

3. running the model on the specified data, 

4. outputting the result
