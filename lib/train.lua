require 'cunn'
require 'cudnn'
require 'optim'
require 'lib/util/eval'

local M = {}
local Trainer = torch.class('image-play.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  assert(opt.seqLength == 2, 'currently supports seqLength = 2 only')
  self.model = model
  self.criterion = criterion
  self.optimState = optimState or {
    learningRate = opt.LR,
    weightDecay = opt.weightDecay
  }
  self.opt = opt
  self.params, self.gradParams = model:getParameters()
  self.logger = {
    train = optim.Logger(paths.concat(opt.save, 'train.log')),
    val = optim.Logger(paths.concat(opt.save, 'val.log'))
  }
end

function Trainer:train(epoch, dataloader)
  local timer = torch.Timer()
  local dataTimer = torch.Timer()

  local function feval()
    return self.criterion.output, self.gradParams
  end

  local size = dataloader:size()

  print('=> Training epoch # ' .. epoch)
  -- Set the batch norm to training mode
  self.model:training()
  for i, sample in dataloader:run() do
    local dataTime = dataTimer:time().real
  
    -- Get input and target
    local input = sample.input
    local target = sample.target

    -- Convert to CUDA
    input, target = self:convertCuda(input, target)
  
    -- Forward pass
    local output = self.model:forward(input)
    local loss = self.criterion:forward(self.model.output, target)
    local loss1 = self.criterion.criterions[1].output
    local loss2 = self.criterion.criterions[2].output

    -- Backprop
    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, target)
    self.model:backward(input, self.criterion.gradInput)
  
    -- Optimization
    optim.rmsprop(feval, self.params, self.optimState)

    -- Compute accuracy
    local acc = {}
    for j = 1, #output do
      acc[j] = self:computeAccuracy(output[j]:contiguous(), target[j])
    end
    acc = torch.Tensor(acc)
    -- Mean accuracy; ignore nan
    local macc
    if torch.all(acc:ne(acc)) then
      macc = 0/0
    else
      macc = torch.mean(acc[acc:eq(acc)])
    end
 
    -- Print and log
    local time = timer:time().real
    print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  ' ..
           'loss1 %1.5f  loss2 %1.5f  acc1 %6.4f  acc2 %6.4f'):format(
               epoch, i, size, time, dataTime, loss1, loss2, acc[1], acc[2]
           )
    )
    self.logger['train']:add{
        ['epoch'] = string.format("%d" % epoch),
        ['iter'] = string.format("%d" % i),
        ['loss1'] = string.format("%.5f" % loss1),
        ['loss2'] = string.format("%.5f" % loss2),
        ['acc1'] = string.format("%.4f" % acc[1]),
        ['acc2'] = string.format("%.4f" % acc[2]),
        ['time'] = string.format("%.3f" % time),
        ['dataTime'] = string.format("%.3f" % dataTime)
    }
  
    timer:reset()
    dataTimer:reset()
  end
end

function Trainer:test(epoch, loaders, split)
  local timer = torch.Timer()
  local dataTimer = torch.Timer()
  local testTimer = torch.Timer()

  assert(split == 'val' or split == 'test',
      'test() only supports validation and test set: to support training ' ..
      'set, check the logging block to make sure training log will not be ' ..
      'messed up'
  )
  local dataloader = loaders[split]
  local size = dataloader:size()
  local lossSum, maccSum = 0.0, 0.0
  local loss1Sum, loss2Sum, acc1Sum, acc2Sum = 0.0, 0.0, 0.0, 0.0
  local N = 0

  self.model:evaluate()
  for i, sample in dataloader:run() do
    local dataTime = dataTimer:time().real

    -- Get input and target
    local input = sample.input
    local target = sample.target

    -- Convert to CUDA
    input, target = self:convertCuda(input, target)

    -- Forward pass
    local output = self.model:forward(input)
    local loss = self.criterion:forward(self.model.output, target)
    local loss1 = self.criterion.criterions[1].output
    local loss2 = self.criterion.criterions[2].output

    -- Compute accuracy
    local acc = {}
    for j = 1, #output do
      acc[j] = self:computeAccuracy(output[j], target[j])
    end
    acc = torch.Tensor(acc)
    -- Mean accuracy; ignore nan
    local macc
    if torch.all(acc:ne(acc)) then
      macc = 0/0
    else
      macc = torch.mean(acc[acc:eq(acc)])
      
      -- local batchSize = input:size(1)
      -- lossSum = lossSum + loss*batchSize
      -- maccSum = maccSum + macc*batchSize
      -- N = N + batchSize
    end
    if torch.all(acc:eq(acc)) then
      local batchSize = input:size(1)
      loss1Sum = loss1Sum + loss1*batchSize
      loss2Sum = loss2Sum + loss2*batchSize
      acc1Sum = acc1Sum + acc[1]*batchSize
      acc2Sum = acc2Sum + acc[2]*batchSize
      N = N + batchSize
    end 

    -- Print and log
    local time = timer:time().real
    print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  ' ..
           'loss1 %1.5f (%1.5f)  loss2 %1.5f (%1.5f)  ' ..
           'acc1 %6.4f (%6.4f)  acc2 %6.4f (%6.4f)'):format(
               epoch, i, size, time, dataTime,
               loss1, loss1Sum / N, loss2, loss2Sum / N,
               acc[1], acc1Sum / N, acc[2], acc2Sum / N
           )
    )

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  local testTime = testTimer:time().real
  print((' * Finished epoch # %d    Time %.3f  ' ..
         'loss1 %1.5f  loss2 %1.5f  acc1 %6.4f  acc2 %6.4f'):format(
         epoch, testTime, loss1Sum / N, loss2Sum / N, acc1Sum / N, acc2Sum / N
         )
  )
  self.logger[split]:add{
      ['epoch'] = string.format("%d" % epoch),
      ['loss1'] = string.format("%.5f" % (loss1Sum / N)),
      ['loss2'] = string.format("%.5f" % (loss2Sum / N)),
      ['acc1'] = string.format("%.4f" % (acc1Sum / N)),
      ['acc2'] = string.format("%.4f" % (acc2Sum / N)),
      ['time'] = string.format("%.3f" % testTime),
  }

  return lossSum / N, maccSum / N
end

function Trainer:convertCuda(input, target)
  input = input:cuda()
  for i = 1, #target do
    target[i] = target[i]:cuda()
  end
  return input, target
end

function Trainer:computeAccuracy(output, target)
  -- Ignore frame with no visible joints
  local keepInd = {}
  for i = 1, target:size(1) do
    if torch.any(target[i]:ne(0)) then
      table.insert(keepInd, i)
    end
  end
  if #keepInd ~= target:size(1) then
    -- Return nan if all frames have no visible joints
    if next(keepInd) == nil then
      return 0/0
    end
    local ind = torch.LongTensor(keepInd)
    output = output:index(1, ind)
    target = target:index(1, ind)
  end

  return heatmapAccuracy(output, target, nil, nil, self.opt.outputRes)
end

return M.Trainer
