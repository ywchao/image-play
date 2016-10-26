require 'cunn'
require 'cudnn'
require 'optim'
require 'lib/util/eval'
require 'lib/util/Logger'
require 'common/util'

local matio = require 'matio'
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
  self.nOutput = #self.model.outnode.children
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
    names[#names+1] = 'err' .. i .. 'p'
  end
  for i = 1, self.opt.seqLength do
    names[#names+1] = 'acc' .. i .. 'p'
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
  self:setSeqLenCritWeight(self.opt.currBase, epoch)

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
  for i, sample in dataloader:run({train=true}) do
    local dataTime = dataTimer:time().real
  
    -- Get input/output and convert to CUDA
    local input = sample.input[1]:cuda()
    local repos, trans, focal, hmap, proj = {}, {}, {}, {}, {}
    for j = 1, #sample.input do
      repos[j] = sample.repos[j]:cuda()
      trans[j] = sample.trans[j]:cuda()
      focal[j] = sample.focal[j]:cuda()
      hmap[j] = sample.hmap[j]:cuda()
      proj[j] = sample.proj[j]:cuda()
    end

    -- Get target
    local target
    if self.nOutput == 1 then target = hmap end

    -- Init output
    local loss, err, acc = {}, {}, {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      local output = self.model:forward(input)
      if self.nOutput == 5 then
        local proj_ = {}
        for j = 1, #proj do
          proj_[j] = proj[j]:clone()
          proj_[j][proj_[j]:eq(0)] = output[5][j][proj_[j]:eq(0)]
        end
        target = {hmap, repos, trans, focal, proj_}
      end
      self.criterion:forward(self.model.output, target)

      -- Backprop
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, target)
      self.model:backward(input, self.criterion.gradInput)

      -- Optimization
      optim.rmsprop(feval, self.params, self.optimState)

      -- Compute loss and error
      for j = 1, self.opt.seqLength do
        proj[j] = proj[j]:float()
        if j <= self.seqlen then
          if self.nOutput == 1 then
            loss[j] = self.criterion.criterions[j].output
            err[j] = self:computeError(getPreds(output[j]:float()),proj[j])
            acc[j] = self:computeAccuracy(getPreds(output[j]:float()),proj[j])
          end
          if self.nOutput == 5 then
            local l1 = self.criterion.criterions[1].criterions[j].output
            local l2 = self.criterion.criterions[5].criterions[j].output
            loss[j] = 0.0
            loss[j] = loss[j] + l1 * 1
            loss[j] = loss[j] + l2 * self.opt.weightProj
            local pred
            if self.opt.evalOut == 's3' then pred = output[5][j]:float() end
            if self.opt.evalOut == 'hg' then pred = getPreds(output[1][j]:float()) end
            err[j] = self:computeError(pred,proj[j])
            acc[j] = self:computeAccuracy(pred,proj[j])
          end
        else
          loss[j] = 0/0
          err[j] = 0/0
          acc[j] = 0/0
        end
      end
      acc = torch.Tensor(acc)
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
      local out_enc, out_rnn, inp_zero, inp_rnn
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
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(inp_zero)
          end
        end
        output[j] = self.model['dec']:forward(out_rnn):clone()
        if j < self.seqlen then
          for _, v in ipairs(self.lstm_ind) do
            self.model['rnn'][j+1].modules[v].hiddenInput = self.model['rnn'][j].modules[v].hiddenOutput
            self.model['rnn'][j+1].modules[v].cellInput = self.model['rnn'][j].modules[v].cellOutput
          end
        end
        self.criterion:forward(self.model['dec'].output, label[j])

        loss[j] = self.criterion.output
        acc[j] = self:computeAccuracy(output[j]:contiguous(), label[j]:float())

        self.criterion:backward(self.model['dec'].output, label[j])
        self.model['dec']:backward(out_rnn, self.criterion.gradInput)
        if torch.type(out_rnn) ~= 'table' then
          gradInputDec[j] = self.model['dec'].gradInput[1]:clone()
        else
          for k = 1, #self.model['dec'].gradInput do
            gradInputDec[j][k] = self.model['dec'].gradInput[k]:clone()
          end
        end
      end
      for j = self.seqlen+1, self.opt.seqLength do
        loss[j] = 0
        acc[j] = 0/0
      end
      acc = torch.Tensor(acc)

      -- Backward pass for RNN and encoder
      local gradInputRNNSum
      for j = self.seqlen, 1, -1 do
        if self.model['res'] then
          if j == 1 then
            inp_rnn = append(out_enc, out_enc)
          else
            inp_rnn = append(out_enc, inp_zero)
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
          for _, v in ipairs(self.lstm_ind) do
            self.model['rnn'][j-1].modules[v].gradHiddenOutput = self.model['rnn'][j].modules[v].gradHiddenInput
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
      entry[#entry+1] = string.format("%7.5f" % loss[j])
    end
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%7.4f" % err[j])
    end
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%7.5f" % acc[j])
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
  local size = dataloader:sizeDataset()
  local lossSum, errSum, accSum = {}, {}, {}
  for i = 1, self.opt.seqLength do
    lossSum[i] = 0.0
    errSum[i] = 0.0
    accSum[i] = 0.0
  end
  local N = 0

  -- Set seqlen if test at iter 0
  if epoch == 0 and iter == 0 then
    self.seqlen = self.opt.seqLength
  end

  print("=> Test on " .. split)
  xlua.progress(0, size)

  self:setModelMode('evaluate')
  for i, sample in dataloader:run({train=false}) do
    -- Get input/output and convert to CUDA
    local input = sample.input[1]:cuda()
    local repos, trans, focal, hmap, proj = {}, {}, {}, {}, {}
    for j = 1, #sample.input do
      repos[j] = sample.repos[j]:cuda()
      trans[j] = sample.trans[j]:cuda()
      focal[j] = sample.focal[j]:cuda()
      hmap[j] = sample.hmap[j]:cuda()
      proj[j] = sample.proj[j]:cuda()
    end

    -- Get target
    local target
    if self.nOutput == 1 then target = hmap end

    -- Init output
    local loss, err, acc = {}, {}, {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      local output = self.model:forward(input)
      if self.nOutput == 5 then
        local proj_ = {}
        for j = 1, #proj do
          proj_[j] = proj[j]:clone()
          proj_[j][proj_[j]:eq(0)] = output[5][j][proj_[j]:eq(0)]
        end
        target = {hmap, repos, trans, focal, proj_}
      end
      self.criterion:forward(self.model.output, target)

      -- Compute loss and error
      for j = 1, self.opt.seqLength do
        proj[j] = proj[j]:float()
        if j <= self.seqlen then
          if self.nOutput == 1 then
            loss[j] = self.criterion.criterions[j].output
            err[j] = self:computeError(getPreds(output[j]:float()),proj[j])
            acc[j] = self:computeAccuracy(getPreds(output[j]:float()),proj[j])
          end
          if self.nOutput == 5 then
            local l1 = self.criterion.criterions[1].criterions[j].output
            local l2 = self.criterion.criterions[5].criterions[j].output
            loss[j] = 0.0
            loss[j] = loss[j] + l1 * 1
            loss[j] = loss[j] + l2 * self.opt.weightProj
            local pred
            if self.opt.evalOut == 's3' then pred = output[5][j]:float() end
            if self.opt.evalOut == 'hg' then pred = getPreds(output[1][j]:float()) end
            err[j] = self:computeError(pred,proj[j])
            acc[j] = self:computeAccuracy(pred,proj[j])
          end
        else
          loss[j] = 0/0
          err[j] = 0/0
          acc[j] = 0/0
        end
      end
      acc = torch.Tensor(acc)
    end

    -- Breakdown nngraph model
    if torch.type(self.model) == 'table' then
      -- Reset RNN States
      self:resetRNNStates()

      -- Forward pass
      local output = {}
      local out_enc, out_rnn, inp_zero, inp_rnn
      for j = 1, self.seqlen do
        if j == 1 then
          out_enc = self.model['enc']:forward(input)
          if torch.type(out_enc) ~= 'table' then
            out_enc = {out_enc}
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, out_enc)
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
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(inp_zero)
          end
        end
        output[j] = self.model['dec']:forward(out_rnn):clone()
        if j < self.seqlen then
          for _, v in ipairs(self.lstm_ind) do
            self.model['rnn'][j+1].modules[v].hiddenInput = self.model['rnn'][j].modules[v].hiddenOutput
            self.model['rnn'][j+1].modules[v].cellInput = self.model['rnn'][j].modules[v].cellOutput
          end
        end
        self.criterion:forward(self.model['dec'].output, label[j])

        loss[j] = self.criterion.output
        acc[j] = self:computeAccuracy(output[j]:contiguous(), label[j]:float())
      end
      for j = self.seqlen+1, self.opt.seqLength do
        loss[j] = 0
        acc[j] = 0/0
      end
      acc = torch.Tensor(acc)
    end

    -- Accumulate loss and acc
    assert(input:size(1) == 1, 'batch size must be 1 with run({train=false})')
    if torch.all(acc:sub(1,self.seqlen):eq(acc:sub(1,self.seqlen))) then
      for j = 1, self.opt.seqLength do
        lossSum[j] = lossSum[j] + loss[j]
        errSum[j] = errSum[j] + err[j]
        accSum[j] = accSum[j] + acc[j]
      end
      N = N + 1
    end 

    xlua.progress(i, size)
  end
  self:setModelMode('training')

  -- Compute mean loss and accuracy
  for i = 1, self.opt.seqLength do
    lossSum[i] = lossSum[i] / N
    errSum[i] = errSum[i] / N
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
    entry[#entry+1] = string.format("%7.5f" % lossSum[j])
  end
  for j = 1, self.opt.seqLength do
    entry[#entry+1] = string.format("%7.4f" % errSum[j])
  end
  for j = 1, self.opt.seqLength do
    entry[#entry+1] = string.format("%7.5f" % accSum[j])
  end
  self.logger[split]:add(entry)
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]

  print("=> Generating predictions ...")
  xlua.progress(0, dataloader:sizeSampled())

  self:setModelMode('evaluate')
  for i, sample in dataloader:run({train=false,samp=true}) do
    -- Get input/output and convert to CUDA
    local index = sample.index
    local input = sample.input[1]:cuda()

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
      local out_enc, out_rnn, inp_zero, inp_rnn
      for j = 1, self.opt.seqLength do
        if j == 1 then
          out_enc = self.model['enc']:forward(input)
          if torch.type(out_enc) ~= 'table' then
            out_enc = {out_enc}
          end
          if self.model['res'] then
            inp_rnn = append(out_enc, out_enc)
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
            out_rnn = self.model['rnn'][j]:forward(inp_rnn)
          else
            out_rnn = self.model['rnn'][j]:forward(inp_zero)
          end
        end
        output[j] = self.model['dec']:forward(out_rnn):clone()
        if j < self.opt.seqLength then
          for _, v in ipairs(self.lstm_ind) do
            self.model['rnn'][j+1].modules[v].hiddenInput = self.model['rnn'][j].modules[v].hiddenOutput
            self.model['rnn'][j+1].modules[v].cellInput = self.model['rnn'][j].modules[v].cellOutput
          end
        end
      end
    end

    -- Save output
    assert(input:size(1) == 1, 'batch size must be 1 with run({train=false})')
    if self.nOutput == 1 then output = {output} end
    
    local pred_path = paths.concat(self.opt.save,'pred_' .. split)
    local pred_file = paths.concat(pred_path, string.format("%05d.mat" % index[1]))
    makedir(pred_path)
    if not paths.filep(pred_file) then
      local hmap = torch.FloatTensor(self.opt.seqLength, output[1][1]:size(2),
          self.opt.outputRes, self.opt.outputRes)
      for j = 1, self.opt.seqLength do hmap[j]:copy(output[1][j][1]) end
      if self.nOutput == 1 then
        matio.save(pred_file, {hmap = hmap})
      end
      if self.nOutput == 5 then
        local repos = torch.FloatTensor(self.opt.seqLength, output[2][1]:size(2), 3)
        local trans = torch.FloatTensor(self.opt.seqLength, 3)
        local focal = torch.FloatTensor(self.opt.seqLength, 1)
        for j = 1, self.opt.seqLength do
          repos[j]:copy(output[2][j][1])
          trans[j]:copy(output[3][j][1])
          focal[j]:copy(output[4][j][1])
        end
        matio.save(pred_file, {hmap = hmap, repos = repos, trans = trans, focal = focal})
      end
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
  self:setModelMode('training')
end

function Trainer:setSeqLenCritWeight(currBase, epoch)
  local seqlen
  -- Start len from 2 and no larger than data sequence length
  -- seqlen = math.ceil(epoch / self.opt.currInt) + 1
  seqlen = currBase * 2 ^ math.ceil((epoch-1) / self.opt.currInt)
  seqlen = math.min(seqlen, self.opt.seqLength)
  -- Single nngraph model
  if torch.type(self.model) == 'nn.gModule' then
    assert(self.nOutput == 1 or self.nOutput == 5)
    if self.nOutput == 1 then
      assert(self.opt.weightHMap == 1)
      for i = 1, self.opt.seqLength do
        if i <= seqlen then
          self.criterion.weights[i] = 1
        else
          self.criterion.weights[i] = 0
        end
      end
    end
    if self.nOutput == 5 then
      for i = 1, self.opt.seqLength do
        self.criterion.criterions[2].weights[i] = 0
        self.criterion.criterions[3].weights[i] = 0
        self.criterion.criterions[4].weights[i] = 0
        if i <= seqlen then
          self.criterion.criterions[1].weights[i] = self.opt.weightHMap
          self.criterion.criterions[5].weights[i] = self.opt.weightProj
        else
          self.criterion.criterions[1].weights[i] = 0
          self.criterion.criterions[5].weights[i] = 0
        end
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

function Trainer:ignoreFrames(output, target)
  -- Ignore frames with no visible joints
  local keepInd = {}
  for i = 1, target:size(1) do
    if torch.any(target[i]:ne(0)) then
      table.insert(keepInd, i)
    end
  end
  if #keepInd ~= target:size(1) then
    -- Return nan if all frames have no visible joints
    if next(keepInd) == nil then
      return 0/0, 0/0
    end
    local ind = torch.LongTensor(keepInd)
    output = output:index(1, ind)
    target = target:index(1, ind)
  end
  return output, target
end

function Trainer:computeError(output, target)
-- target: N x d x 2
-- output: N x d x 2
  local output, target = self:ignoreFrames(output, target)
  if output ~= output then
    return 0/0
  end
  local e, n = {}, {}
  for i = 1, target:size(1) do
    e[i], n[i] = 0.0, 0.0
    for j = 1, target:size(2) do
      if target[i][j][1] > 0 then
        local p1 = target:select(2,j)
        local p2 = output:select(2,j)
        n[i] = n[i] + 1
        e[i] = e[i] + torch.csub(p1,p2):pow(2):sum(2):sqrt()[1][1]
      end
    end
  end
  return torch.cdiv(torch.Tensor(e),torch.Tensor(n)):mean()
end

function Trainer:computeAccuracy(output, target)
-- target: N x d x 2
-- output: N x d x 2
  local output, target = self:ignoreFrames(output, target)
  if output ~= output then
    return 0/0
  end
  return coordAccuracy(output, target, nil, nil, self.opt.outputRes)
end

return M.Trainer