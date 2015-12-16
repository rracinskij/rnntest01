-- This is a modification of an example provided at https://github.com/Element-Research/rnn#rnn.Recurrent
require 'rnn'
fun = require 'fun'
require 'gnuplot'

-- Dataset which is a cyclic sequence of the digits [1, 9]. Given a digit the
-- model should learn to predict the subsequent digit.
nLength  = 20
sequence = fun.range(1, 9):cycle():take(nLength):totable()

print("Sequence:", table.concat(sequence, ", "))
print("")

rho        = 5
hiddenSize = 10
outputSize = 1  -- Next predicted digit

-- Configuration of the model
model = nn.Sequential()
model:add(nn.Recurrent(
  -- Output size of the RNN
  hiddenSize,

  -- Input layer: Use a lookup table containing `nLength` tensors of size
  -- `hiddenSize`.
  nn.LookupTable(nLength, hiddenSize),

  -- Feedback layer: Recurrence; use nn.Linear to work with numbers < 1
  nn.Linear(hiddenSize, hiddenSize),

  -- Transfer function
  nn.Sigmoid(),

  -- Maximum number of time steps for BPTT, default is 9999 meaning that the
  -- changes will be backpropgated through the entire sequence.
  rho
))
model:add(nn.Linear(hiddenSize, outputSize))

-- Criterion: Mean squared error
criterion = nn.MSECriterion()

-- For comparison, use a different criterion
-- criterion = nn.SmoothL1Criterion()

function updateGradient(model, x, y, criterion, learningRate, iteration)
  local prediction  = model:forward(x)

  -- Use criterion to compute the loss and its gradients
  local error       = criterion:forward (prediction, y)
  local gradOutputs = criterion:backward(prediction, y)

  -- The recurrent layer is memorising its gradOutputs
  model:backward(x, gradOutputs)

  -- Update interval must be < rho
  local updateInterval = 3

  -- Backpropagation through time (BPTT)
  if iteration % updateInterval == 0 then
    -- Backward through feedback and input layers
    model:backwardThroughTime()

    -- Update parameters
    model:updateParameters(learningRate)

    -- Zero the accumulation of the gradients with respect to model parameters
    model:zeroGradParameters()

    -- Reset the internal time-step counter
    model:forget()
  end

  return prediction, error
end

learningRate = 0.1
epochs       = 50

-- For each epoch iterate over the entire sequence
epochErrors = fun.range(1, epochs):map(function (epoch)
  print("Epoch " .. epoch)

  local errors = fun.range(1, nLength - 1):map(function (i)
    local input  = torch.Tensor(1):fill(sequence[i])
    local target = torch.Tensor(1):fill(sequence[i + 1])  -- Next number in sequence
    local prediction, error =
      updateGradient(model, input, target, criterion, learningRate, i)

    print(
      "Input: ", input[1],
      " Target: ", target[1],
      " Output: ", prediction[1][1],
      " Error: ", error)

    return error
  end)

  local avgError = errors:sum() / errors:length()

  print("Average error: ", avgError)
  print("")

  return avgError
end)

gnuplot.title('RNN')
gnuplot.xlabel('Epoch')
gnuplot.plot({'Average error', torch.Tensor(epochErrors:totable()), '+-'})
gnuplot.grid(true)
gnuplot.axis({
  0, 50,  -- Limits of x axis
  0, 1    -- Limits of y axis
})
