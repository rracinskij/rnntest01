-- This is a modification of an example provided at https://github.com/Element-Research/rnn#rnn.Recurrent
require 'rnn'

--batchSize = 1
rho = 5
hiddenSize = 10
nIndex = 20
-- RNN
r = nn.Recurrent(
   hiddenSize, --size of the input layer
   nn.LookupTable(nIndex, hiddenSize), --input layer (inputSize, outputSize) (use nn.Linear to work with numbers <1)
   nn.Linear(hiddenSize, hiddenSize), --recurrent layer
   nn.Sigmoid(), --transfer function
   rho  --maximum number of time steps for BPTT
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, 1))--nIndex))
--rnn:add(nn.LogSoftMax()) 

criterion = nn.MSECriterion() 

-- dummy dataset (task is to predict next item, given previous)
i = 0
sequence = torch.Tensor(nIndex):apply(function() --fill with a simple arithmetic progression 1,2..9,1,2...
  i = i + 1
  if i == 10 then i = 1 end
  return i
end)
print('Sequence:')
print(sequence)

lr = 0.1
step = 0
threshold = 0.002
thresholdStep = 0
--while true do
for k = 1, 100 do
for j = 1, 9 do
   step = step + 1
   -- a batch of inputs
   local input = torch.Tensor(1):fill(sequence[j])
   local output = rnn:forward(input)
   local target = torch.Tensor(1):fill(sequence[j+1]) --target is the next numbet in sequence
   local err = criterion:forward(output, target)
   print('Step: ', step, ' Input: ', input[1], ' Target: ', target[1], ' Output: ', output[1][1], ' Error: ', err)
   if (err < threshold and thresholdStep == 0) then thresholdStep = step end --remember this step
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)
   
   -- note that updateInterval < rho
   if j % 3 == 0 then --update interval
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      rnn:backwardThroughTime()
      -- 2. updates parameters
      rnn:updateParameters(lr)
      rnn:zeroGradParameters()
      -- 3. reset the internal time-step counter
      rnn:forget()
   end --end if
end -- end j
end -- end k

print('Error < ', threshold,' on step: ', thresholdStep)
