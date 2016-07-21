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

function Trainer:train(epoch, loaders)
  local timer = torch.Timer()
  local dataTimer = torch.Timer()

  local function feval()
    return self.criterion.output, self.gradParams
  end

  local dataloader = loaders['train']
  local size = dataloader:size()
  local testint = self.opt.testInt
  local ntest = math.ceil(size / testint)
  local itest = 1
  local goal

  print(('=> Training epoch # %d [%d/%d]'):format(epoch, itest, ntest))
  goal = (math.min((itest * testint), size) - 1) % testint + 1
  xlua.progress(0, goal)

  -- Set the batch norm to training mode
  self.model:training()
  for i, sample in dataloader:run({augment=true}) do
    local dataTime = dataTimer:time().real
  
    -- Get input and target
    local input = sample.input[1]
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

    xlua.progress(((i - 1) % testint) + 1, goal)

    -- Test on validation
    if i % testint == 0 or i == size then
      self:test(epoch, i, loaders, 'val')

      if i ~= size then
        itest = itest + 1
        print(('=> Training epoch # %d [%d/%d]'):format(epoch, itest, ntest))
        goal = (math.min((itest * testint), size) - 1) % testint + 1
        xlua.progress(0, goal)
      end
    end
  end
end

function Trainer:test(epoch, iter, loaders, split)
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

  print("=> Test on " .. split)
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run() do
    -- Get input and target
    local input = sample.input[1]
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

    xlua.progress(i, size)
  end
  self.model:training()

  local testTime = testTimer:time().real
  self.logger[split]:add{
      ['epoch'] = string.format("%d" % epoch),
      ['iter'] = string.format("%d" % iter),
      ['loss1'] = string.format("%.5f" % (loss1Sum / N)),
      ['loss2'] = string.format("%.5f" % (loss2Sum / N)),
      ['acc1'] = string.format("%.4f" % (acc1Sum / N)),
      ['acc2'] = string.format("%.4f" % (acc2Sum / N)),
      ['time'] = string.format("%.3f" % testTime),
  }

  return lossSum / N, maccSum / N
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local heatmaps

  print("=> Generating predictions ...")
  xlua.progress(0, dataloader:sizeDataset())

  self.model:evaluate()
  for i, sample in dataloader:run({pred=true}) do
    -- Get input and target
    local index = sample.index
    local input = sample.input[1]
    local target = sample.target

    -- Convert to CUDA
    input, target = self:convertCuda(input, target)

    -- Forward pass
    local output = self.model:forward(input)

    -- Copy output
    if not heatmaps then
      local outputDim = target[1]:size(2)
      heatmaps = torch.Tensor(
          dataloader:sizeDataset(), self.opt.seqLength,
          outputDim, self.opt.outputRes, self.opt.outputRes
      )
    end
    assert(input:size(1) == 1, 'batch size must be 1 with run({pred=true})')
    for j = 1, #index do
      for k = 1, #output do
        -- fill(0) to heatmaps is too slow
        -- assert(torch.all(heatmaps[index[j]][k]:eq(0)),
        --     'overwriting heatmap in heatmaps')
        heatmaps[index[j]][k]:copy(output[k][j])
      end
    end

    xlua.progress(i, dataloader:sizeDataset())
  end
  self.model:training()

  -- Save final predictions
  local f = hdf5.open(self.opt.save .. '/preds_' .. split .. '.h5', 'w')
  f:write('heatmaps', heatmaps)
  f:close()
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
