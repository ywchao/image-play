require 'cunn'
require 'cudnn'
require 'optim'
require 'lib/util/eval'
require 'lib/util/Logger'

local M = {}
local Trainer = torch.class('image-play.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  self.model = model
  self.criterion = criterion
  self.optimState = optimState or {
    learningRate = opt.LR,
    weightDecay = opt.weightDecay
  }
  self.opt = opt
  self.params, self.gradParams = model:getParameters()
  self.logger = {
    train = Logger(paths.concat(opt.save, 'train.log'), opt.resume),
    val = Logger(paths.concat(opt.save, 'val.log'), opt.resume)
  }
  self:initLogger(self.logger['train'])
  self:initLogger(self.logger['val'])
end

function Trainer:initLogger(logger)
  local names = {}
  names[1] = 'epoch'
  names[2] = 'iter'
  names[3] = 'time'
  names[4] = 'datTime'
  for i = 1, self.opt.seqLength do
    names[#names+1] = 'loss' .. i
  end
  for i = 1, self.opt.seqLength do
    names[#names+1] = 'acc' .. i
  end
  logger:setNames(names)
end

function Trainer:train(epoch, loaders)
  local timer = torch.Timer()
  local dataTimer = torch.Timer()

  local function feval()
    return self.criterion.output, self.gradParams
  end

  -- Set criterion weight for curriculum learning
  self:setCriterionWeight(epoch)

  local dataloader = loaders['train']
  local size = dataloader:size()
  local testint = self.opt.testInt
  local ntest = math.ceil(size / testint)
  local itest = 1
  local goal

  print(('=> Training epoch # %d [%d/%d], seq %d'):format(
            epoch, itest, ntest, self.seqlen
         )
  )
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
    local loss_sum = self.criterion:forward(self.model.output, target)
    local loss = {}
    for j = 1, #output do
      if j <= self.seqlen then
        loss[j] = self.criterion.criterions[j].output
      else
        loss[j] = 0
      end
    end

    -- Backprop
    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, target)
    self.model:backward(input, self.criterion.gradInput)
  
    -- Optimization
    optim.rmsprop(feval, self.params, self.optimState)

    -- Compute accuracy
    local acc = {}
    for j = 1, #output do
      if j <= self.seqlen then
        acc[j] = self:computeAccuracy(output[j]:contiguous(), target[j])
      else
        acc[j] = 0/0
      end
    end
    acc = torch.Tensor(acc)

    -- Print and log
    local time = timer:time().real
    local entry = {}
    entry[1] = string.format("%d" % epoch)
    entry[2] = string.format("%d" % i)
    entry[3] = string.format("%.3f" % time)
    entry[4] = string.format("%.3f" % dataTime)
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % loss[j])
    end
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % acc[j])
    end
    self.logger['train']:add(entry)
  
    timer:reset()
    dataTimer:reset()

    xlua.progress(((i - 1) % testint) + 1, goal)

    -- Test on validation
    if i % testint == 0 or i == size then
      self:test(epoch, i, loaders, 'val')

      if i ~= size then
        itest = itest + 1
        print(('=> Training epoch # %d [%d/%d], seq %d'):format(
                   epoch, itest, ntest, self.seqlen
               )
        )
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
  local lossSum, accSum = {}, {}
  for i = 1, self.opt.seqLength do
    lossSum[i] = 0.0
    accSum[i] = 0.0
  end
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
    local loss_sum = self.criterion:forward(self.model.output, target)
    local loss = {}
    for j = 1, #output do
      if j <= self.seqlen then
        loss[j] = self.criterion.criterions[j].output
      else
        loss[j] = 0
      end
    end

    -- Compute accuracy
    local acc = {}
    for j = 1, #output do
      if j <= self.seqlen then
        acc[j] = self:computeAccuracy(output[j], target[j])
      else
        acc[j] = 0/0
      end
    end
    acc = torch.Tensor(acc)
    if torch.all(acc:sub(1,self.seqlen):eq(acc:sub(1,self.seqlen))) then
      local batchSize = input:size(1)
      for j = 1, self.opt.seqLength do
        lossSum[j] = lossSum[j] + loss[j]
        accSum[j] = accSum[j] + acc[j]
      end
      N = N + batchSize
    end 

    xlua.progress(i, size)
  end
  self.model:training()

  -- Compute mean loss and accuracy
  for i = 1, self.opt.seqLength do
    lossSum[i] = lossSum[i] / N
    accSum[i] = accSum[i] / N
  end

  -- Print and log
  local testTime = testTimer:time().real
  local entry = {}
  entry[1] = string.format("%d" % epoch)
  entry[2] = string.format("%d" % iter)
  entry[3] = string.format("%.3f" % testTime)
  entry[4] = string.format("%d" % 0/0)
  for j = 1, self.opt.seqLength do
    entry[#entry+1] = string.format("%.5f" % lossSum[j])
  end
  for j = 1, self.opt.seqLength do
    entry[#entry+1] = string.format("%.5f" % accSum[j])
  end
  self.logger[split]:add(entry)
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local sidx = torch.LongTensor(dataloader:sizeSampled())
  local heatmaps

  print("=> Generating predictions ...")
  xlua.progress(0, dataloader:sizeSampled())

  self.model:evaluate()
  for i, sample in dataloader:run({pred=true,samp=true}) do
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
      heatmaps = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          outputDim, self.opt.outputRes, self.opt.outputRes
      )
    end
    assert(input:size(1) == 1, 'batch size must be 1 with run({pred=true})')
    sidx[i] = index[1]
    for j = 1, #output do
      heatmaps[i][j]:copy(output[j][1])
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
  self.model:training()

  -- Sort heatmaps by index
  local sidx, i = torch.sort(sidx)
  heatmaps = heatmaps:index(1, i)

  -- Save final predictions
  local f = hdf5.open(self.opt.save .. '/preds_' .. split .. '.h5', 'w')
  f:write('heatmaps', heatmaps)
  f:close()
end

function Trainer:setCriterionWeight(epoch)
  local seqlen
  -- Start len from 2 and no larger than data sequence length
  -- seqlen = math.ceil(epoch / self.opt.currInt) + 1
  seqlen = 2 ^ math.ceil(epoch / self.opt.currInt)
  seqlen = math.min(seqlen, self.opt.seqLength)
  for i = 1, #self.criterion.weights do
    if i <= seqlen then
      self.criterion.weights[i] = 1
    else
      self.criterion.weights[i] = 0
    end
  end
  self.seqlen = seqlen
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
