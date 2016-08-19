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
  if #model:findModules('nn.SplitTable') == 2 then
    self.predFl = true
  else
    self.predFl = false
  end
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
    names[#names+1] = 'loss' .. i .. 'p'
  end
  for i = 1, self.opt.seqLength do
    names[#names+1] = 'acc' .. i .. 'p'
  end
  if self.predFl then
    for i = 1, self.opt.seqLength do
      names[#names+1] = 'loss' .. i .. 'f'
    end
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
    local target_ps = sample.target_ps
    local target_fl = sample.target_fl

    -- Convert to CUDA
    input, target_ps, target_fl = self:convertCuda(input, target_ps, target_fl)
  
    -- Forward pass
    local output = self.model:forward(input)
    local loss_ps, loss_fl = {}, {}
    local acc_ps = {}
    if self.predFl then
      self.criterion:forward(self.model.output, {target_ps, target_fl})
      for j = 1, #output[1] do
        if j <= self.seqlen then
          loss_ps[j] = self.criterion.criterions[1].criterions[j].output
          loss_fl[j] = self.criterion.criterions[2].criterions[j].output
          acc_ps[j] = self:computeAccuracy(output[1][j]:contiguous(), target_ps[j])
        else
          loss_ps[j] = 0
          loss_fl[j] = 0
          acc_ps[j] = 0/0
        end
      end
    else
      self.criterion:forward(self.model.output, target_ps)
      for j = 1, #output do
        if j <= self.seqlen then
          loss_ps[j] = self.criterion.criterions[j].output
          acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        else
          loss_ps[j] = 0
          acc_ps[j] = 0/0
        end
        loss_fl[j] = 0/0
      end
    end
    acc_ps = torch.Tensor(acc_ps)

    -- Backprop
    self.model:zeroGradParameters()
    if self.predFl then
      self.criterion:backward(self.model.output, {target_ps, target_fl})
    else
      self.criterion:backward(self.model.output, target_ps)
    end
    self.model:backward(input, self.criterion.gradInput)
  
    -- Optimization
    optim.rmsprop(feval, self.params, self.optimState)

    -- Print and log
    local time = timer:time().real
    local entry = {}
    entry[1] = string.format("%d" % epoch)
    entry[2] = string.format("%d" % i)
    entry[3] = string.format("%.3f" % time)
    entry[4] = string.format("%.3f" % dataTime)
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % loss_ps[j])
    end
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % acc_ps[j])
    end
    if self.predFl then
      for j = 1, self.opt.seqLength do
        entry[#entry+1] = string.format("%.5f" % loss_fl[j])
      end
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
  local lossSum_ps, lossSum_fl, accSum_ps = {}, {}, {}
  for i = 1, self.opt.seqLength do
    lossSum_ps[i] = 0.0
    lossSum_fl[i] = 0.0
    accSum_ps[i] = 0.0
  end
  local N = 0

  -- Set seqlen if test at iter 0
  if epoch == 0 and iter == 0 then
    self.seqlen = self.opt.seqLength
  end

  print("=> Test on " .. split)
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run() do
    -- Get input and target
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local target_fl = sample.target_fl

    -- Convert to CUDA
    input, target_ps, target_fl = self:convertCuda(input, target_ps, target_fl)

    -- Forward pass
    local output = self.model:forward(input)
    local loss_ps, loss_fl = {}, {}
    local acc_ps = {}
    if self.predFl then
      self.criterion:forward(self.model.output, {target_ps, target_fl})
      for j = 1, #output[1] do
        if j <= self.seqlen then
          loss_ps[j] = self.criterion.criterions[1].criterions[j].output
          loss_fl[j] = self.criterion.criterions[2].criterions[j].output
          acc_ps[j] = self:computeAccuracy(output[1][j]:contiguous(), target_ps[j])
        else
          loss_ps[j] = 0
          loss_fl[j] = 0
          acc_ps[j] = 0/0
        end
      end
    else
      self.criterion:forward(self.model.output, target_ps)
      for j = 1, #output do
        if j <= self.seqlen then
          loss_ps[j] = self.criterion.criterions[j].output
          acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        else
          loss_ps[j] = 0
          acc_ps[j] = 0/0
        end
        loss_fl[j] = 0/0
      end
    end
    acc_ps = torch.Tensor(acc_ps)

    -- Accumulate loss and acc
    if torch.all(acc_ps:sub(1,self.seqlen):eq(acc_ps:sub(1,self.seqlen))) then
      local batchSize = input:size(1)
      for j = 1, self.opt.seqLength do
        lossSum_ps[j] = lossSum_ps[j] + loss_ps[j]
        lossSum_fl[j] = lossSum_fl[j] + loss_fl[j]
        accSum_ps[j] = accSum_ps[j] + acc_ps[j]
      end
      N = N + batchSize
    end 

    xlua.progress(i, size)
  end
  self.model:training()

  -- Compute mean loss and accuracy
  for i = 1, self.opt.seqLength do
    lossSum_ps[i] = lossSum_ps[i] / N
    lossSum_fl[i] = lossSum_fl[i] / N
    accSum_ps[i] = accSum_ps[i] / N
  end

  -- Print and log
  local testTime = testTimer:time().real
  local entry = {}
  entry[1] = string.format("%d" % epoch)
  entry[2] = string.format("%d" % iter)
  entry[3] = string.format("%.3f" % testTime)
  entry[4] = string.format("%d" % 0/0)
  for j = 1, self.opt.seqLength do
    entry[#entry+1] = string.format("%.5f" % lossSum_ps[j])
  end
  for j = 1, self.opt.seqLength do
    entry[#entry+1] = string.format("%.5f" % accSum_ps[j])
  end
  if self.predFl then
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % lossSum_fl[j])
    end
  end
  self.logger[split]:add(entry)
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local sidx = torch.LongTensor(dataloader:sizeSampled())
  local heatmaps, flows, gtflows

  print("=> Generating predictions ...")
  xlua.progress(0, dataloader:sizeSampled())

  self.model:evaluate()
  for i, sample in dataloader:run({pred=true,samp=true}) do
    -- Get input and target
    local index = sample.index
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local target_fl = sample.target_fl

    -- Convert to CUDA
    input, target_ps, target_fl = self:convertCuda(input, target_ps, target_fl)

    -- Forward pass
    local output = self.model:forward(input)

    -- Copy output
    if not heatmaps then
      heatmaps = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          target_ps[1]:size(2), self.opt.outputRes, self.opt.outputRes
      )
      if self.predFl then
        flows = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          2, self.opt.outputRes, self.opt.outputRes
        )
        gtflows = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          2, self.opt.outputRes, self.opt.outputRes
        )
      end
    end
    assert(input:size(1) == 1, 'batch size must be 1 with run({pred=true})')
    sidx[i] = index[1]
    if self.predFl then
      for j = 1, #output[1] do
        heatmaps[i][j]:copy(output[1][j][1])
        -- heatmaps[i][j]:copy(target_ps[2][j][1])
        -- Get flow and target_fl
        flows[i][j]:copy(output[2][j][1])
        gtflows[i][j]:copy(target_fl[j][1])
      end
    else
      for j = 1, #output do
        heatmaps[i][j]:copy(output[j][1])
        -- heatmaps[i][j]:copy(target_ps[j][1])
      end
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
  self.model:training()

  -- Sort heatmaps by index
  local sidx, i = torch.sort(sidx)
  heatmaps = heatmaps:index(1, i)
  if self.predFl then
    flows = flows:index(1, i)
    gtflows = gtflows:index(1, i)
  end

  -- Save final predictions
  local f = hdf5.open(self.opt.save .. '/preds_' .. split .. '.h5', 'w')
  -- local f = hdf5.open(self.opt.save .. '/gt_' .. split .. '.h5', 'w')
  f:write('heatmaps', heatmaps)
  f:close()

  -- Save flow separately
  if self.predFl then
    local f = hdf5.open(self.opt.save .. '/flows_' .. split .. '.h5', 'w')
    f:write('flows', flows)
    f:write('gtflows', gtflows)
    f:close()
  end
end

function Trainer:setCriterionWeight(epoch)
  local seqlen
  -- Start len from 2 and no larger than data sequence length
  -- seqlen = math.ceil(epoch / self.opt.currInt) + 1
  seqlen = 2 ^ math.ceil(epoch / self.opt.currInt)
  seqlen = math.min(seqlen, self.opt.seqLength)
  if self.predFl then
    for i = 1, #self.criterion.criterions[1].weights do
      if i <= seqlen then
        self.criterion.criterions[1].weights[i] = 1
        self.criterion.criterions[2].weights[i] = 1
      else
        self.criterion.criterions[1].weights[i] = 0
        self.criterion.criterions[2].weights[i] = 0
      end
    end
  else
    for i = 1, #self.criterion.weights do
      if i <= seqlen then
        self.criterion.weights[i] = 1
      else
        self.criterion.weights[i] = 0
      end
    end
  end
  self.seqlen = seqlen
end

function Trainer:convertCuda(input, target_ps, target_fl)
  input = input:cuda()
  for i = 1, #target_ps do
    target_ps[i] = target_ps[i]:cuda()
    target_fl[i] = target_fl[i]:cuda()
  end
  return input, target_ps, target_fl
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
