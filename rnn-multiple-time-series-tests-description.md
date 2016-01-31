This is a brief desctription of torch-rnn multiple time series example testing results.
Original example - https://github.com/Element-Research/rnn/blob/master/examples/recurrent-time-series.lua
Modified code for testing purposes - https://github.com/rracinskij/rnntest01/blob/master/rnn-multiple-time-series-tests.lua

Dataset is created synthetically and contains three variables:

Column 1: a 0.01, 0.02, 0.03, ... 0.4, 0.01, 0.02 ... sequence

Column 2: a 0.01, 0.01, 0.02, 0.03, 0.05, .. 0.55, 0.01, 0.01 (Fibonacci divided by 100) sequence

Column 3: calculated as Column1 * 0.5 - Column2 * 0.5 + Column3 * 0.3 in t=-1

Each test was performed with two models:
- reccurent model with a linear recurrent layer from the original example with rho = 3, number of hidden units = 8 and Tanh activation function;
- LSTM model with rho = 3 and 8 hidden units.

Test 1:
Input: Column 1 in t=-1, -2,...
Target: Column 1 in t=0


