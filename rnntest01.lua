-- Simple torch-rnn demo (https://github.com/Element-Research/rnn)
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
for i=3,nIndex do --create a Fibonnaci sequence 1,1,2,3,5,8,13,...; If the input is 1, output can be 1 or 2 depending on previous input
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
   nn.LookupTable(nIndex, hiddenSize), --input layer
   nn.Linear(hiddenSize, hiddenSize), --recurrent layer
   nn.Tanh(), 
   rho
)

local rnn = nn.Sequential()
   :add(r)    
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.LogSoftMax()) --classifier

-- add sequencer
rnn = nn.Sequencer(rnn)

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

print("model:", rnn)

-- training
local iteration = 1
local seqIndex = 1
while iteration<100 do

   -- 1. create a sequence of rho time-steps
   if seqIndex > nIndex-rho then seqIndex = 1 end --continue from beginning after end of sequence is reached
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

   -- retrieve output, since LogSoftMax returns log(p)
       local out = outputs[rho]:clone():exp() --copy the tensor, or operation will be performed on initial tensor otherwise
       maxIndex = 0
       local maxOutput = 0 
       for i=1,nIndex do
         if out[1][i]>maxOutput then
           maxOutput = out[1][i]
           maxIndex = i
          end
        end 
      print('# iteration: ', iteration, 'input:', inputs[rho][1], 'target:', targets[rho][1], 'output:', maxIndex)

   -- 3. backward sequence through rnn (i.e. backprop through time)
   rnn:zeroGradParameters()
   local gradOutputs = criterion:backward(outputs, targets) 
   local gradInputs = rnn:backward(inputs, gradOutputs) 
   -- note that LookupTable does not generate any gradInputs and it can be a problem in more complicated models. 
   -- a workaround is in https://github.com/Element-Research/rnn/issues/185

   -- 4. update
   rnn:updateParameters(lr)
   
   iteration = iteration + 1
  
end --end iteration
