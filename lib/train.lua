require 'cunn'
require 'cudnn'
require 'optim'
require 'lib/util/eval'
require 'lib/util/Logger'

local model_utils = require 'lib/util/model_utils'

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
  -- Single nngraph model
  if torch.type(model) == 'nn.gModule' then
    self.params, self.gradParams = model:getParameters()
  end
  -- Breakdown nngraph model
  if torch.type(model) == 'table' then
    self.params, self.gradParams = model_utils.combine_all_parameters(
        model['enc'], model['rnn_one'], model['dec']
    )
    self.model['rnn'] = model_utils.clone_many_times(
        model['rnn_one'], self.opt.seqLength
    )
    self.lstm_ind = {}
    for i, m in ipairs(model['rnn'][1].modules) do
      if torch.type(m) == 'cudnn.LSTM' then
        table.insert(self.lstm_ind, i)
      end
    end
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
  if self.model['res'] and self.model['vae'] then
    for i = 1, self.opt.seqLength do
      names[#names+1] = 'loss' .. i .. 'z'
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
  self:setSeqLenCritWeight(epoch)

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
  self:setModelMode('training')
  for i, sample in dataloader:run({augment=true}) do
    local dataTime = dataTimer:time().real
  
    -- Get input and target
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local epsilon = sample.epsilon

    -- Convert to CUDA
    input, target_ps, epsilon = self:convertCuda(input, target_ps, epsilon)

    -- Init output
    local loss_ps, loss_z = {}, {}
    local acc_ps = {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      local output = self.model:forward(input)
      self.criterion:forward(self.model.output, target_ps)
      for j = 1, #output do
        if j <= self.seqlen then
          loss_ps[j] = self.criterion.criterions[j].output
          acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        else
          loss_ps[j] = 0
          acc_ps[j] = 0/0
        end
      end
      acc_ps = torch.Tensor(acc_ps)

      -- Backprop
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, target_ps)
      self.model:backward(input, self.criterion.gradInput)

      -- Optimization
      optim.rmsprop(feval, self.params, self.optimState)
    end

    -- Breakdown nngraph model
    if torch.type(self.model) == 'table' then
      -- Reset RNN States
      self:resetRNNStates()

      -- Zero gradient params
      self:zeroGradParams()

      collectgarbage()

      --- Forward pass and decoder backward pass
      local output = {}
      local out_enc, out_rnn, inp_zero, inp_rnn, inp_dec
      local gradInputDec = {}
      for j = 1, self.seqlen do
        gradInputDec[j] = {}
      end
      for j = 1, self.seqlen do
        if j == 1 then
          out_enc = self.model['enc']:forward(input)
          if torch.type(out_enc) ~= 'table' then
            out_enc = {out_enc}
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, out_enc)
            if self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(out_enc)
          end
        else
          if inp_zero == nil then
            inp_zero = {}
            for k = 1, #out_enc do
              inp_zero[k] = torch.zeros(out_enc[k]:size()):cuda()
            end
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, inp_zero)
            if self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(inp_zero)
          end
        end
        if self.model['res'] and self.model['vae'] then
          inp_dec = slice(out_rnn, #out_enc)
        else
          inp_dec = out_rnn
        end
        if #inp_dec == 1 then
          inp_dec = inp_dec[1]
        end
        output[j] = self.model['dec']:forward(inp_dec):clone()
        if j < self.seqlen then
          for k, v in ipairs(self.lstm_ind) do
            if self.model['res'] and self.model['vae'] then
              self.model['rnn'][j+1].modules[v].hiddenInput = out_rnn[#out_enc+k]
            else
              self.model['rnn'][j+1].modules[v].hiddenInput = self.model['rnn'][j].modules[v].hiddenOutput
            end
            self.model['rnn'][j+1].modules[v].cellInput = self.model['rnn'][j].modules[v].cellOutput
          end
        end
        self.criterion:forward(self.model['dec'].output, target_ps[j])

        loss_ps[j] = self.criterion.output
        acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        if self.model['res'] and self.model['vae'] then
          loss_z[j] = torch.mean(out_rnn[#out_enc*2+1])
        else
          loss_z[j] = 0/0
        end

        self.criterion:backward(self.model['dec'].output, target_ps[j])
        self.model['dec']:backward(out_rnn, self.criterion.gradInput)
        if torch.type(self.model['dec'].gradInput) ~= 'table' then
          self.model['dec'].gradInput = {self.model['dec'].gradInput}
        end
        if torch.type(out_rnn) ~= 'table' then
          gradInputDec[j] = self.model['dec'].gradInput[1]:clone()
        else
          for k = 1, #self.model['dec'].gradInput do
            gradInputDec[j][k] = self.model['dec'].gradInput[k]:clone()
          end
        end
      end
      for j = self.seqlen+1, self.opt.seqLength do
        loss_ps[j] = 0
        acc_ps[j] = 0/0
        loss_z[j] = 0/0
      end
      acc_ps = torch.Tensor(acc_ps)

      -- Backward pass for RNN and encoder
      local gradInputRNNSum, gradInputZ, gradInputLossZ
      if self.model['res'] and self.model['vae'] then
        gradInputZ, gradInputLossZ = {}, {}
        for j = 1, #out_enc do
          gradInputZ[j] = torch.zeros(out_rnn[#out_enc+j]:size()):cuda()
        end
        local batchSize = out_rnn[#out_enc*2+1]:numel()
        gradInputLossZ[1] = torch.ones(batchSize):div(batchSize):cuda()
      end
      for j = self.seqlen, 1, -1 do
        if self.model['res'] then
          if j == 1 then
            inp_rnn = append(out_enc, out_enc)
            if self.model['res'] and self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
          else
            inp_rnn = append(out_enc, inp_zero)
            if self.model['res'] and self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
          end
          if self.model['res'] and self.model['vae'] then
            gradInputDec[j] = append(gradInputDec[j], gradInputZ)
            gradInputDec[j] = append(gradInputDec[j], gradInputLossZ)
          end
          self.model['rnn'][j]:backward(inp_rnn, gradInputDec[j])
          if j == self.seqlen then
            gradInputRNNSum = {}
            for k = 1, #out_enc do
              gradInputRNNSum[k] = self.model['rnn'][j].gradInput[k]:clone()
            end
          else
            for k = 1, #out_enc do
              gradInputRNNSum[k] = gradInputRNNSum[k] + self.model['rnn'][j].gradInput[k]
            end
          end
          -- Need to propagate gradient back from RNN for the first frame
          -- Spent two days to figure this out
          if j == 1 then
            for k = 1, #out_enc do
              gradInputRNNSum[k] = gradInputRNNSum[k] + self.model['rnn'][j].gradInput[k+#out_enc]
            end
          end
        else
          if j ~= 1 then
            self.model['rnn'][j]:backward(inp_zero, gradInputDec[j])
          else
            self.model['rnn'][j]:backward(out_enc, gradInputDec[j])
          end
        end
        if j ~= 1 then
          for k, v in ipairs(self.lstm_ind) do
            if self.model['res'] and self.model['vae'] then
              gradInputZ[k] = self.model['rnn'][j].modules[v].gradHiddenInput
            else
              self.model['rnn'][j-1].modules[v].gradHiddenOutput = self.model['rnn'][j].modules[v].gradHiddenInput
            end
            self.model['rnn'][j-1].modules[v].gradCellOutput = self.model['rnn'][j].modules[v].gradCellInput
          end
        end
      end

      -- Backward pass for encoder
      if #gradInputRNNSum == 1 then
        gradInputRNNSum = gradInputRNNSum[1]
      end
      if self.model['res'] then
        self.model['enc']:backward(input, gradInputRNNSum)
      else
        self.model['enc']:backward(input, self.model['rnn'][1].gradInput)
      end

      -- Optimization
      optim.rmsprop(feval, self.params, self.optimState)
    end

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
    if self.model['res'] and self.model['vae'] then
      for j = 1, self.opt.seqLength do
        entry[#entry+1] = string.format("%.8f" % loss_z[j])
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
  local lossSum_ps, accSum_ps, lossSum_z = {}, {}, {}
  for i = 1, self.opt.seqLength do
    lossSum_ps[i] = 0.0
    lossSum_z[i] = 0.0
    accSum_ps[i] = 0.0
  end
  local N = 0

  -- Set seqlen if test at iter 0
  if epoch == 0 and iter == 0 then
    self.seqlen = self.opt.seqLength
  end

  print("=> Test on " .. split)
  xlua.progress(0, size)

  self:setModelMode('evaluate')
  for i, sample in dataloader:run() do
    -- Get input and target
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local epsilon = sample.epsilon

    -- Convert to CUDA
    input, target_ps, epsilon = self:convertCuda(input, target_ps, epsilon)

    -- Init output
    local loss_ps, loss_z = {}, {}
    local acc_ps = {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      local output = self.model:forward(input)
      self.criterion:forward(self.model.output, target_ps)
      for j = 1, #output do
        if j <= self.seqlen then
          loss_ps[j] = self.criterion.criterions[j].output
          acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        else
          loss_ps[j] = 0
          acc_ps[j] = 0/0
        end
      end
      acc_ps = torch.Tensor(acc_ps)
    end

    -- Breakdown nngraph model
    if torch.type(self.model) == 'table' then
      -- Reset RNN States
      self:resetRNNStates()

      -- Forward pass
      local output = {}
      local out_enc, out_rnn, inp_zero, inp_rnn, inp_dec
      for j = 1, self.seqlen do
        if j == 1 then
          out_enc = self.model['enc']:forward(input)
          if torch.type(out_enc) ~= 'table' then
            out_enc = {out_enc}
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, out_enc)
            if self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(out_enc)
          end
        else
          if inp_zero == nil then
            inp_zero = {}
            for k = 1, #out_enc do
              inp_zero[k] = torch.zeros(out_enc[k]:size()):cuda()
            end
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, inp_zero)
            if self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(inp_zero)
          end
        end
        if self.model['res'] and self.model['vae'] then
          inp_dec = slice(out_rnn, #out_enc)
        else
          inp_dec = out_rnn
        end
        if #inp_dec == 1 then
          inp_dec = inp_dec[1]
        end
        output[j] = self.model['dec']:forward(inp_dec):clone()
        if j < self.seqlen then
          for k, v in ipairs(self.lstm_ind) do
            if self.model['res'] and self.model['vae'] then
              self.model['rnn'][j+1].modules[v].hiddenInput = out_rnn[#out_enc+k]
            else
              self.model['rnn'][j+1].modules[v].hiddenInput = self.model['rnn'][j].modules[v].hiddenOutput
            end
            self.model['rnn'][j+1].modules[v].cellInput = self.model['rnn'][j].modules[v].cellOutput
          end
        end
        self.criterion:forward(self.model['dec'].output, target_ps[j])

        loss_ps[j] = self.criterion.output
        acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        if self.model['res'] and self.model['vae'] then
          loss_z[j] = torch.mean(out_rnn[#out_enc*2+1])
        else
          loss_z[j] = 0/0
        end
      end
      for j = self.seqlen+1, self.opt.seqLength do
        loss_ps[j] = 0
        acc_ps[j] = 0/0
        loss_z[j] = 0/0
      end
      acc_ps = torch.Tensor(acc_ps)
    end

    -- Accumulate loss and acc
    if torch.all(acc_ps:sub(1,self.seqlen):eq(acc_ps:sub(1,self.seqlen))) then
      local batchSize = input:size(1)
      for j = 1, self.opt.seqLength do
        lossSum_ps[j] = lossSum_ps[j] + loss_ps[j]
        lossSum_z[j] = lossSum_z[j] + loss_z[j]
        accSum_ps[j] = accSum_ps[j] + acc_ps[j]
      end
      N = N + batchSize
    end 

    xlua.progress(i, size)
  end
  self:setModelMode('training')

  -- Compute mean loss and accuracy
  for i = 1, self.opt.seqLength do
    lossSum_ps[i] = lossSum_ps[i] / N
    lossSum_z[i] = lossSum_z[i] / N
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
  if self.model['res'] and self.model['vae'] then
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.8f" % lossSum_z[j])
    end
  end
  self.logger[split]:add(entry)
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local sidx = torch.LongTensor(dataloader:sizeSampled())
  local heatmaps

  print("=> Generating predictions ...")
  xlua.progress(0, dataloader:sizeSampled())

  self:setModelMode('evaluate')
  for i, sample in dataloader:run({pred=true,samp=true}) do
    -- Get input and target
    local index = sample.index
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local epsilon = sample.epsilon

    -- Convert to CUDA
    input, target_ps, epsilon = self:convertCuda(input, target_ps, epsilon)

    -- Init output
    local output = {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      output = self.model:forward(input)
    end

    -- Breakdown nngraph model
    if torch.type(self.model) == 'table' then
      -- Reset RNN States
      self:resetRNNStates()

      -- Forward pass
      local out_enc, out_rnn, inp_zero, inp_rnn, inp_dec
      for j = 1, self.opt.seqLength do
        if j == 1 then
          out_enc = self.model['enc']:forward(input)
          if torch.type(out_enc) ~= 'table' then
            out_enc = {out_enc}
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, out_enc)
            if self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(out_enc)
          end
        else
          if inp_zero == nil then
            inp_zero = {}
            for k = 1, #out_enc do
              inp_zero[k] = torch.zeros(out_enc[k]:size()):cuda()
            end
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, inp_zero)
            if self.model['vae'] then
              inp_rnn = append(inp_rnn, epsilon[j])
            end
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(inp_zero)
          end
        end
        if self.model['res'] and self.model['vae'] then
          inp_dec = slice(out_rnn, #out_enc)
        else
          inp_dec = out_rnn
        end
        if #inp_dec == 1 then
          inp_dec = inp_dec[1]
        end
        output[j] = self.model['dec']:forward(inp_dec):clone()
        if j < self.opt.seqLength then
          for k, v in ipairs(self.lstm_ind) do
            if self.model['res'] and self.model['vae'] then
              self.model['rnn'][j+1].modules[v].hiddenInput = out_rnn[#out_enc+k]
            else
              self.model['rnn'][j+1].modules[v].hiddenInput = self.model['rnn'][j].modules[v].hiddenOutput
            end
            self.model['rnn'][j+1].modules[v].cellInput = self.model['rnn'][j].modules[v].cellOutput
          end
        end
      end
    end

    -- Copy output
    if not heatmaps then
      heatmaps = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          target_ps[1]:size(2), self.opt.outputRes, self.opt.outputRes
      )
    end
    assert(input:size(1) == 1, 'batch size must be 1 with run({pred=true})')
    sidx[i] = index[1]
    for j = 1, #output do
      heatmaps[i][j]:copy(output[j][1])
      -- heatmaps[i][j]:copy(target_ps[j][1])
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
  self:setModelMode('training')

  -- Sort heatmaps by index
  local sidx, i = torch.sort(sidx)
  heatmaps = heatmaps:index(1, i)

  -- Save final predictions
  local f = hdf5.open(self.opt.save .. '/preds_' .. split .. '.h5', 'w')
  -- local f = hdf5.open(self.opt.save .. '/gt_' .. split .. '.h5', 'w')
  f:write('heatmaps', heatmaps)
  f:close()
end

function Trainer:setSeqLenCritWeight(epoch)
  local seqlen
  -- Start len from 2 and no larger than data sequence length
  -- seqlen = math.ceil(epoch / self.opt.currInt) + 1
  seqlen = 2 ^ math.ceil(epoch / self.opt.currInt)
  seqlen = math.min(seqlen, self.opt.seqLength)
  -- Single nngraph model
  if torch.type(self.model) == 'nn.gModule' then
    for i = 1, #self.criterion.weights do
      if i <= seqlen then
        self.criterion.weights[i] = 1
      else
        self.criterion.weights[i] = 0
      end
    end
  end
  -- breakdown nngraph model
  if torch.type(self.model) == 'table' then
    -- No need to set criterion weight
  end
  self.seqlen = seqlen
end

function Trainer:setModelMode(mode)
  assert(mode == 'training' or mode == 'evaluate')
  if mode == 'training' then
    if torch.type(self.model) == 'nn.gModule' then
      self.model:training()
    end
    if torch.type(self.model) == 'nn.table' then
      self.model['enc']:training()
      self.model['dec']:training()
      for i = 1, #self.model['rnn'] do
        self.model['rnn'][i]:training()
      end
    end
  end
  if mode == 'evaluate' then
    if torch.type(self.model) == 'nn.gModule' then
      self.model:evaluate()
    end
    if torch.type(self.model) == 'nn.table' then
      self.model['enc']:evaluate()
      self.model['dec']:evaluate()
      for i = 1, #self.model['rnn'] do
        self.model['rnn'][i]:evaluate()
      end
    end
  end
end

function Trainer:convertCuda(input, target_ps, epsilon)
  input = input:cuda()
  for i = 1, #target_ps do
    target_ps[i] = target_ps[i]:cuda()
  end
  for i = 1, #epsilon do
    for j = 1, #epsilon[i] do
      epsilon[i][j] = epsilon[i][j]:cuda()
    end
  end
  return input, target_ps, epsilon
end

function Trainer:resetRNNStates()
  for i = 1, #self.model['rnn'] do
    self.model['rnn'][i]:apply(function(module)
      if torch.type(module) == 'cudnn.LSTM' then
        module:resetStates()
      end
    end)
  end
end

function Trainer:zeroGradParams()
  self.model['enc']:zeroGradParameters()
  self.model['dec']:zeroGradParameters()
  for i = 1, #self.model['rnn'] do
    self.model['rnn'][i]:zeroGradParameters()
  end
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
