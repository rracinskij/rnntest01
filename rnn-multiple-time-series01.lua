--this is a modification of the rnn-multiple time series example of Torch-rnn
-- Multi-variate time-series test
-- based on torch-rnn example

require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a multivariate time-series model using RNN')
cmd:option('--rho', 5, 'maximum number of time steps for back-propagate through time (BPTT)')
cmd:option('--multiSize', 2, 'number of random variables as input and output')
cmd:option('--hiddenSize', 12, 'number of hidden units used at output of the recurrent layer')
cmd:option('--dataSize', 100, 'total number of time-steps in dataset')
cmd:option('--batchSize', 8, 'number of training samples per batch')
cmd:option('--nIterations', 1000, 'max number of training iterations')
cmd:option('--learningRate', 0.05, 'learning rate')
cmd:text()
local opt = cmd:parse(arg or {})


sequence = torch.Tensor(opt.dataSize,opt.multiSize+1):fill(0)
i=0
for j = 2,opt.dataSize do
sequence[j][1]= i
sequence[j][2] = i + 0.01
sequence[j][3] = sequence[j-1][1] + sequence[j-1][2]
i=i+0.01
if sequence[j][3] >0.99 then i=0 end
end
print('Sequence:'); print(sequence)
print('Sequence length:', sequence:size(1))


-- batch mode

--offsets = torch.LongTensor(opt.batchSize):random(1,opt.dataSize)

j=0
offsets = torch.LongTensor(opt.batchSize):apply(function()
    j=j+1
    return j
  end)
print('offsets: ', offsets)

-- RNN
r = nn.Recurrent(
   opt.hiddenSize, -- size of output
   nn.Linear(opt.multiSize, opt.hiddenSize), -- input layer
   --nn.Linear(opt.hiddenSize, opt.hiddenSize), -- recurrent layer
   nn.LSTM(opt.hiddenSize, opt.hiddenSize),
   nn.Tanh(), -- transfer function
   opt.rho
)

rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(opt.hiddenSize, 1))

criterion = nn.MSECriterion() 

-- use Sequencer for better data handling
rnn = nn.Sequencer(rnn)

criterion = nn.SequencerCriterion(criterion)
print("Model :")
print(rnn)

-- train rnn model
minErr = opt.multiSize -- report min error
minK = 0
avgErrs = torch.Tensor(opt.nIterations):fill(0)
for k = 1, opt.nIterations do 

   -- 1. create a sequence of rho time-steps
   
   local inputs, targets = {}, {}
   for step = 1, opt.rho do
      -- batch of inputs
      inputs[step] = inputs[step] or sequence.new()
      inputs[step]:index(sequence, 1, offsets)
      inputs[step] = inputs[step]:sub(1,8,1,2)
      -- batch of targets
      offsets:add(1) -- increase indices by 1
      offsets[offsets:gt(sequence:size(1))] = 1
      targets[step] = targets[step] or sequence.new()
      targets[step]:index(sequence, 1, offsets)
      targets[step] = targets[step]:sub(1,8,3,3)
   end

   -- 2. forward sequence through rnn

   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
  
   if k%100==0 then 
   for i=1,opt.rho do
      print('iter: ', k, 'element: ', i)
      print('inputs:')
      print(inputs[i])
      print('targets:')
      print(targets[i])
      print('outputs:')
      print(outputs[i])
   end --end for i
   end --end if
   
   -- report errors
 
   print('Iter: ' .. k .. '   Err: ' .. err)
   avgErrs[k] = err
   if avgErrs[k] < minErr then
      minErr = avgErrs[k]
      minK = k
   end

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   rnn:zeroGradParameters()
   
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = rnn:backward(inputs, gradOutputs)

   -- 4. updates parameters
   
   rnn:updateParameters(opt.learningRate)
end

print('min err: ' .. minErr .. ' on iteration ' .. minK)
