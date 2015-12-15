-- This is a modification of an example provided at https://github.com/Element-Research/rnn#rnn.Recurrent
require 'rnn'
fun = require 'fun'

-- Dataset which is a cyclic sequence of the digits [1, 9]. Given a digit the
-- model should learn to predict the subsequent digit.
nLength          = 20
sequenceIterator = fun.range(1, 9):cycle():take(nLength)

print('Sequence:')
sequenceIterator:each(print)

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

function gradientUpgrade(model, x, y, criterion, learningRate, i, j)
  local prediction = model:forward(x)
  local err = criterion:forward(prediction, y)
  local gradOutputs = criterion:backward(prediction, y)

  -- The recurrent layer is memorizing its gradOutputs (up to memSize)
  model:backward(x, gradOutputs)

  -- Update interval must be < rho
  local updateInterval = 3

  -- Backpropagation through time (BPTT)
  if j % updateInterval == 0 then
    -- Backward through feedback and input layers
    model:backwardThroughTime()

    -- Update parameters
    model:updateParameters(learningRate)

    -- Zero the accumulation of the gradients with respect to model parameters
    model:zeroGradParameters()

    -- Reset the internal time-step counter
    model:forget()
  end

  return prediction, err
end

sequence     = sequenceIterator:totable()
learningRate = 0.1
iterations   = 50
threshold    = 0.002

-- Step where the error rate was lower than `threshold`
step          = 1
thresholdStep = 0

-- For each iteration iterate over the entire sequence
for i = 1, iterations do
  for j = 1, nLength - 1 do
    local input  = torch.Tensor(1):fill(sequence[j])
    local target = torch.Tensor(1):fill(sequence[j + 1])  -- Next number in sequence
    local prediction, error = gradientUpgrade(model, input, target, criterion, learningRate, i, j)

    print('Step: ', step, ' Input: ', input[1], ' Target: ', target[1], ' Output: ', prediction[1][1], ' Error: ', error)
    if (error < threshold and thresholdStep == 0) then thresholdStep = step end

    step = step + 1
  end
end

print('Error < ', threshold,' on step: ', thresholdStep)
