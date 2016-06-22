-- Simple torch-rnn (https://github.com/Element-Research/rnn) demo
-- based on torch-rnn library demos
require 'rnn'

batchSize = 1
rho = 3 --number of time steps
hiddenSize = 120
inputSize = 1
nIndex = 100 --LookupTable input space
lr = 0.2 --learning rate

-- build a dummy dataset (task is to predict next item, given previous)
sequence = torch.Tensor(nIndex):fill(1)
for i=3,nIndex do --create a Fibonnaci sequence 1,1,2,3,5,8,13,...; If the input is 1, target is either 1 or 2 depending on previous step
   sequence[i]=sequence[i-1]+sequence[i-2]
   if sequence[i] > nIndex then 
  sequence[i] = 1
  sequence[i-1] = 1
  end
end

print('Sequence:')
print(sequence)

-- define model
-- recurrent layer
local r = nn.Recurrent(
   hiddenSize, --output size
   nn.LookupTable(nIndex, hiddenSize), --input layer. Use discrete space to apply LookupTable (https://github.com/Element-Research/rnn/issues/113)
   nn.Linear(hiddenSize, hiddenSize), --recurrent layer
   nn.Tanh(), 
   rho
)

local rnn = nn.Sequential()
   :add(r) --add recurrent layer
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.LogSoftMax()) --classifier

-- add sequencer
rnn = nn.Sequencer(rnn)
--set criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

print("model:", rnn)

-- training
local iteration = 1
local seqIndex = 1
while iteration<100 do

   -- 1. create a sequence of rho time-steps
   if seqIndex > nIndex-rho then seqIndex = 1 end
   local inputs, targets = {}, {}
   for step=1,rho do
      inputs[step] = sequence:sub(seqIndex,seqIndex) --select input
      targets[step] = sequence:sub(seqIndex+1,seqIndex+1) --select target
      seqIndex = seqIndex + 1
   end
   seqIndex = seqIndex - rho+1
   
   -- 2. forward sequence through rnn
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   -- get the classifier output
   maxOutput, maxIndex = torch.max(outputs[rho][1],1) 
   print('# iteration: ', iteration, 'input:', inputs[rho][1], 'target:', targets[rho][1], 'output:', maxIndex[1])

   -- 3. backward sequence through rnn (i.e. backprop through time)
   rnn:zeroGradParameters()
   local gradOutputs = criterion:backward(outputs, targets) 
   local gradInputs = rnn:backward(inputs, gradOutputs) 
   -- note that LookupTable does not generate any gradInputs and it can be a problem in more complicated models. 
   -- please refer to https://github.com/Element-Research/rnn/issues/185 for a workaround

   -- 4. update
   rnn:updateParameters(lr)
   
   iteration = iteration + 1
  
end --end iteration
