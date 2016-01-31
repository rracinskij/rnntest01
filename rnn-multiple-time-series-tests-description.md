This is a brief desctription of torch-rnn multiple time series example testing results.
Original example - https://github.com/Element-Research/rnn/blob/master/examples/recurrent-time-series.lua
Modified code for testing purposes - https://github.com/rracinskij/rnntest01/blob/master/rnn-multiple-time-series-tests.lua

Dataset is created synthetically and contains three variables:

Column 1: a 0.01, 0.02, 0.03, ... 0.4, 0.01, 0.02 ... sequence

Column 2: a 0.01, 0.01, 0.02, 0.03, 0.05, .. 0.55, 0.01, 0.01 (Fibonacci divided by 100) sequence

Column 3: calculated as Column1 * 0.5 - Column2 * 0.5 + Column3 * 0.3 in t=-1

Each test was performed with two models:
- recurrent model with a linear recurrent layer from the original example with rho = 3, number of hidden units = 8 and Tanh activation function;
- LSTM model with rho = 3 and 8 hidden units.

Number of iterations: 990

Learning rate: 0.05

Test 01:
Input: Column 1 in t=-1, -2,...
Target: Column 1 in t=0
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

Test 01:
Input: Column 1 in t=-1, -2,...
Target: Column 1 in t=0
Model: LSTM
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
 0.2886
 0.2927
 0.2968
 0.3008
 0.3047
 0.3086
 0.3125
 0.3163

Err: 0.0078843542625712



