-- Multi-variate time-series test
-- modification of the original recurrent-time-series.lua example

require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a multivariate time-series model using RNN')
cmd:option('--rho', 3, 'maximum number of time steps for back-propagate through time (BPTT)')
cmd:option('--hiddenSize', 8, 'number of hidden units used at output of the recurrent layer')
cmd:option('--dataSize', 100, 'total number of time-steps in dataset')
cmd:option('--batchSize', 8, 'number of training samples per batch')
cmd:option('--nIterations', 990, 'max number of training iterations')
cmd:option('--learningRate', 0.05, 'learning rate')
cmd:option('--inputFirst', 1, 'first index of input variables, 1 to 3')
cmd:option('--inputLast', 1, 'last index of input variables, 1 to 3')
cmd:option('--outputFirst', 1, 'first index of output variables, 1 to 3')
cmd:option('--outputLast', 1, 'last index of output variables, 1 to 3')
cmd:option('--model', 1, 'training model (1 - linear, 2 -LSTM)')
cmd:text()
local opt = cmd:parse(arg or {})

-- create an inputs/outputs sequence
sequence = torch.Tensor(opt.dataSize,3):fill(0.01)
sequence[1][1]=0.02
sequence[2][1]=0.01
for j = 3,opt.dataSize do
--0.01,0.02,...0.40,0.01.. sequence in column 1
sequence[j][1]= sequence[j-1][1]+0.01 
if sequence[j][1] > 0.40 then sequence[j][1] = 0.01 end

--0.01,0.01,0.02,0.03,0.05,0.08... sequence in column 2
sequence[j][2]= sequence[j-1][2]+sequence[j-2][2] 
if sequence[j][2] > 1 then 
  sequence[j][2] = 0.01
  sequence[j-1][2] = 0.01
end --end if

--sequence 3 
sequence[j][3] = sequence[j-1][1]*0.5-sequence[j-1][2]*0.5+sequence[j-1][3]*0.3

end --end for
print('Sequence:'); print(sequence)
print('Sequence length:', sequence:size(1))


-- batch mode

-- create linear offsets for better readability
j=0
offsets = torch.LongTensor(opt.batchSize):apply(function()
    j=j+1
    return j
  end)
print('offsets: ', offsets)


-- RNN
r = nn.Recurrent(
   opt.hiddenSize,
   nn.Linear(opt.inputLast-opt.inputFirst+1, opt.hiddenSize), -- input layer
   nn.Linear(opt.hiddenSize, opt.hiddenSize), -- recurrent layer
   --nn.LSTM(opt.hiddenSize, opt.hiddenSize), -- recurrent layer
   nn.Tanh(), -- transfer function
   opt.rho
)
rnn1 = nn.Sequential()
   :add(r)
   :add(nn.Linear(opt.hiddenSize, opt.outputLast-opt.outputFirst+1)) 
   
rnn2 = nn.Sequential()
    :add(nn.Linear(opt.inputLast-opt.inputFirst+1, opt.hiddenSize))
    :add(nn.LSTM(opt.hiddenSize, opt.hiddenSize))
    :add(nn.Linear(opt.hiddenSize, opt.outputLast-opt.outputFirst+1))
    :add(nn.Tanh())
 

criterion = nn.MSECriterion() 

-- use Sequencer for better data handling
rnn1 = nn.Sequencer(rnn1)
rnn2 = nn.Sequencer(rnn2)

criterion = nn.SequencerCriterion(criterion)


if opt.model == 2 then model = rnn2 else model = rnn1 end
print("Model:")
print(model)


-- train rnn model
minErr = 0 -- report min error
minK = 0
avgErrs = torch.Tensor(opt.nIterations):fill(0)
for k = 1, opt.nIterations do 

   -- 1. create a sequence of rho time-steps
   
   local inputs, targets = {}, {}
   for step = 1, opt.rho do
      -- batch of inputs
      inputs[step] = inputs[step] or sequence.new()
      inputs[step]:index(sequence, 1, offsets)
      inputs[step] = inputs[step]:sub(1,opt.batchSize,opt.inputFirst,opt.inputLast) --select inputs from given columns
      -- batch of targets
      offsets:add(1) -- increase indices by 1
      offsets[offsets:gt(opt.dataSize)] = 1
      targets[step] = targets[step] or sequence.new()
      targets[step]:index(sequence, 1, offsets)
      targets[step] = targets[step]:sub(1,opt.batchSize,opt.outputFirst,opt.outputLast) --select output from given columns
   end -- end for step

   -- 2. forward sequence through rnn

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, targets)
  
   if k%33==0 then 
   for i=1,opt.rho do
      print('iter: ', k, 'element: ', i)
      print('inputs:')
      print(inputs[i])
      print('targets:')
      print(targets[i])
      print('outputs:')
      print(outputs[i])
   end --end for i
   end --end if k
   
   -- report errors
 
   if k%33 == 0 then print('Iter: ' .. k .. '   Err: ' .. err) end
   avgErrs[k] = err
   if avgErrs[k] < minErr then
      minErr = avgErrs[k]
      minK = k
   end

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   model:zeroGradParameters()
   
   local gradOutputs = criterion:backward(outputs, targets)
   local gradInputs = model:backward(inputs, gradOutputs)

   -- 4. updates parameters
   
   model:updateParameters(opt.learningRate)
end

print('min err: ' .. minErr .. ' on iteration ' .. minK)
