Long Short Term Memory Networks
===============================

Can learn time series with long lags between events.
A part of the state of the art deep learning models for recognition of sequences eg: speech or handwriting
A deep RNN network that can contain several LSTM layers stacked on each other

Four inputs -- 

Three Gates 
 - Output Gate
 - Input Gate
 - Forget Gate controls the previous state of the cell impacting on the next state

sigmoid or tanh(full signal or zero energy)

LSTM Layer with Projection

Step 1 : Pre-nonlinear signals calculation

h_{t-1} : initialize with some constant, say 0.1
Weight matrix 
Step 2 : Non-linearity
 - Input and forgetting state e.g : sigmoid
 - Cell State update e.g :  Hyperbolic tangent
 - output signal update e.g : element wise product/projection matrix(mapping from one dimension to another dimension) 
    
    
  

95% of the time is spent on computing the matrix product

Need to optimize this Matrix Multiplication:
Low-rank approximation with SVD
SVD: Factorize the weight matrix as,
    W = U * E * V
    U and v are unitary matrices
    E is a diagonal matrix with singular values in descending order
   
Complexity Analysis of LRA
  Number of multiple with feature vector
  Before Low-Rank Approximation
  
LRA Impact on Accuracy
  LRA results in slight degradation in accuracy
  
Quantization of matrices
  Many quantization schemes exist
  Basic idea is to represent 4 byte floats as 2 or 1 byte integers
  
  ???
  
  
