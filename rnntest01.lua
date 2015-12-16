-- This is a modification of an example provided at https://github.com/Element-Research/rnn#rnn.Recurrent
require 'rnn'
fun = require 'fun'
require 'gnuplot'
require 'optim'

-- Dataset which is a cyclic sequence of the digits [1, 9]. Given a digit the
-- model should learn to predict the subsequent digit.
local nLength  = 20
local sequence = fun.range(1, 9):cycle():take(nLength):totable()

print("Sequence:", table.concat(sequence, ", "))
print("")

-- Criterion: Mean squared error
local criterion = nn.MSECriterion()

-- For comparison, use a different criterion
-- criterion = nn.SmoothL1Criterion()

function forwardBackwardPass(model, x, y, criterion)
  local prediction  = model:forward(x)

  -- Use criterion to compute the loss and its gradients
  local loss        = criterion:forward (prediction, y)
  local gradOutputs = criterion:backward(prediction, y)

  -- The recurrent layer is memorising its gradOutputs
  model:backward(x, gradOutputs)

  return prediction, loss
end

function updateParametersManual(model, x, y, criterion, learningRate, iteration)
  local prediction, loss = forwardBackwardPass(model, x, y, criterion)

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

  return prediction[1], loss
end

local sgdState = {}
function updateParametersSGD(model, input, target, criterion, learningRate, iteration)
  -- Obtain weights and gradients from model
  local modelParams, modelGradParams = model:getParameters()

  local sgdParams = {
    learningRate = learningRate,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
  }

  local prediction = {}

  -- Compute value of the loss function at `input` and its gradient
  local eval = function(newModelParams)
    -- Set the new weights in the model if they have changed
    if modelParams ~= newModelParams then modelParams:copy(newModelParams) end

    -- Reset gradients; gradients are always accumulated to accommodate batch
    -- methods.
    modelGradParams:zero()

    local _prediction, loss =
      forwardBackwardPass(model, input, target, criterion)
    prediction[1] = _prediction

    return loss, modelGradParams
  end

  local _, fs = optim.sgd(eval, modelParams, sgdParams, sgdState)
  local loss = fs[1]

  return prediction[1][1], loss
end

function createModel()
  local rho        = 5
  local hiddenSize = 10
  local outputSize = 1  -- Next predicted digit

  -- Configuration of the model
  local model = nn.Sequential()
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

  return model
end

function train(model, updateParameters)
  local learningRate = 0.1
  local epochs       = 100

  -- For each epoch iterate over the entire sequence
  local losses = fun.range(1, epochs):map(function (epoch)
    print("Epoch " .. epoch)

    local losses = fun.range(1, nLength - 1):map(function (i)
      local input  = torch.Tensor(1):fill(sequence[i])
      local target = torch.Tensor(1):fill(sequence[i + 1])
      local prediction, loss =
        updateParameters(model, input, target, criterion, learningRate, i)

      print(
        "Input: ", input[1],
        " Target: ", target[1],
        " Output: ", prediction[1],
        " Loss: ", loss)

      return loss
    end)

    local avgLoss = losses:sum() / losses:length()

    print("Average error: ", avgLoss)
    print("")

    return avgLoss
  end)

  return torch.Tensor(losses:totable())
end

local manualLosses = train(createModel(), updateParametersManual)
local sgdLosses    = train(createModel(), updateParametersSGD)

gnuplot.title('RNN')
gnuplot.xlabel('Epoch')
gnuplot.plot(
  {'Average loss (manual)', manualLosses, '+-'},
  {'Average loss (SGD)', sgdLosses, '+-'})
gnuplot.grid(true)
gnuplot.axis({
  0, 100,  -- Limits of x axis
  0, 1     -- Limits of y axis
})
