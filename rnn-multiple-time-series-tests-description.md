This is a brief description of torch-rnn multiple time series example testing results.
Original example - https://github.com/Element-Research/rnn/blob/master/examples/recurrent-time-series.lua
Modified code for testing purposes - https://github.com/rracinskij/rnntest01/blob/master/rnn-multiple-time-series-tests.lua

Dataset is created synthetically and contains three variables:

Column 1: a 0.01, 0.02, 0.03, ... 0.4, 0.01, 0.02 ... sequence

Column 2: a 0.01, 0.01, 0.02, 0.03, 0.05, .. 0.55, 0.01, 0.01 (Fibonacci divided by 100) sequence

Column 3: calculated as Column1 * 0.5 - Column2 * 0.5 + Column3 * 0.3 in t=-1

Each test was performed with a recurrent model with a linear recurrent layer from the original example with rho = 3, number of hidden units = 8 and Tanh activation function. LSTM model with rho = 3 and 8 hidden units is included as option.

Number of iterations: 990

Learning rate: 0.05

Test 01: (0.01, 0.02, 0.03... sequence)

Input: Variable 1 in t=-1, -2,... 

Target: Variable 1 in t=0

Model: linear

Last iteration:

inputs:
 0.3000
 0.3100
 0.3200
 0.3300
 0.3400
 0.3500
 0.3600
 0.3700

targets:	
 0.3100
 0.3200
 0.3300
 0.3400
 0.3500
 0.3600
 0.3700
 0.3800

outputs:	
 0.3050
 0.3073
 0.3095
 0.3118
 0.3141
 0.3163
 0.3186
 0.3208

Err: 0.0023547501232523

Test 02: (0.01, 0.01, 0.02, 0.03, 0.05 sequence)

Input: Variable 2 in t=-1, -2,...

Target: Variable 2 in t=0

Model: linear

Last iteration:

inputs:
 0.5500
 0.0100
 0.0100
 0.0200
 0.0300
 0.0500
 0.0800
 0.1300

targets:
 0.0100
 0.0100
 0.0200
 0.0300
 0.0500
 0.0800
 0.1300
 0.2100

outputs:
 0.2939
 0.0252
 0.0847
 0.1329
 0.1359
 0.1424
 0.1519
 0.1677

Err: 0.070459260975717

Test 03: (all three sequences)

Input: Variables 1-3 in t=-1, -2,...

Target: Variables 1-3 in t=0

Model: linear

Last iteration:

inputs:	
 0.3000  0.5500 -0.0046
 
 0.3100  0.0100 -0.1264
 
 0.3200  0.0100  0.1121
 
 0.3300  0.0200  0.1886
 
 0.3400  0.0300  0.2116
 
 0.3500  0.0500  0.2185
 
 0.3600  0.0800  0.2155
 
 0.3700  0.1300  0.2047
 

targets:	
 0.3100  0.0100 -0.1264
 
 0.3200  0.0100  0.1121
 
 0.3300  0.0200  0.1886
 
 0.3400  0.0300  0.2116
 
 0.3500  0.0500  0.2185
 
 0.3600  0.0800  0.2155
 
 0.3700  0.1300  0.2047
 
 0.3800  0.2100  0.1814
 

outputs:	
 0.2517  0.2859 -0.1069
 
 0.2065  0.0651  0.1033
 
 0.2793  0.1330  0.1595
 
 0.3489  0.1632  0.2172
 
 0.3538  0.1673  0.2252
 
 0.3573  0.1743  0.2219
 
 0.3590  0.1835  0.2106
 
 0.3579  0.1980  0.1867
 

Err: 0.028445357200187


