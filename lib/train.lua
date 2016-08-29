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
    if #model:findModules('nn.SplitTable') == 2 then
      local stab_ind = {}
      for i, m in ipairs(model.modules) do
        if torch.type(m) == 'nn.SplitTable' then
          table.insert(stab_ind, i)
        end
      end
      assert(#stab_ind == 2)
      assert(torch.type(model.modules[stab_ind[1]-2]) == 'cudnn.SpatialConvolution')
      if model.modules[stab_ind[2]-2].nOutputPlane == 3 then
        self.predIm = true
        self.predFl = false
      end
      if model.modules[stab_ind[2]-2].nOutputPlane == 2 then
        self.predIm = false
        self.predFl = true
      end
    else
      self.predIm = false
      self.predFl = false
    end
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
    self.predIm = false
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
  if self.predIm and not self.predFl then
    for i = 1, self.opt.seqLength do
      names[#names+1] = 'loss' .. i .. 'i'
    end
  end
  if not self.predIm and self.predFl then
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
    local target_im = sample.target_im
    local target_fl = sample.target_fl

    -- Convert to CUDA
    input, target_ps, target_im, target_fl =
        self:convertCuda(input, target_ps, target_im, target_fl)

    -- Init output
    local loss_ps, loss_im, loss_fl = {}, {}, {}
    local acc_ps = {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      local output = self.model:forward(input)
      if self.predIm and not self.predFl then
        self.criterion:forward(self.model.output, {target_ps, target_im})
        for j = 1, #output[1] do
          if j <= self.seqlen then
            loss_ps[j] = self.criterion.criterions[1].criterions[j].output
            loss_im[j] = self.criterion.criterions[2].criterions[j].output
            acc_ps[j] = self:computeAccuracy(output[1][j]:contiguous(), target_ps[j])
          else
            loss_ps[j] = 0
            loss_im[j] = 0
            acc_ps[j] = 0/0
          end
          loss_fl[j] = 0/0
        end
      end
      if not self.predIm and self.predFl then
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
          loss_im[j] = 0/0
        end
      end
      if not self.predIm and not self.predFl then
        self.criterion:forward(self.model.output, target_ps)
        for j = 1, #output do
          if j <= self.seqlen then
            loss_ps[j] = self.criterion.criterions[j].output
            acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
          else
            loss_ps[j] = 0
            acc_ps[j] = 0/0
          end
          loss_im[j] = 0/0
          loss_fl[j] = 0/0
        end
      end
      acc_ps = torch.Tensor(acc_ps)

      -- Backprop
      self.model:zeroGradParameters()
      if self.predIm and not self.predFl then
        self.criterion:backward(self.model.output, {target_ps, target_im})
      end
      if not self.predIm and self.predFl then
        self.criterion:backward(self.model.output, {target_ps, target_fl})
      end
      if not self.predIm and not self.predFl then
        self.criterion:backward(self.model.output, target_ps)
      end
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
        self.criterion:forward(self.model['dec'].output, target_ps[j])

        loss_ps[j] = self.criterion.output
        acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        -- loss_im[j] = 0/0
        -- loss_fl[j] = 0/0

        self.criterion:backward(self.model['dec'].output, target_ps[j])
        self.model['dec']:backward(out_rnn, self.criterion.gradInput)
        for k = 1, #self.model['dec'].gradInput do
          gradInputDec[j][k] = self.model['dec'].gradInput[k]:clone()
        end
      end
      for j = self.seqlen+1, self.opt.seqLength do
        loss_ps[j] = 0
        acc_ps[j] = 0/0
        -- loss_im[j] = 0/0
        -- loss_fl[j] = 0/0
      end
      acc_ps = torch.Tensor(acc_ps)

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
    if self.predIm and not self.predFl then
      for j = 1, self.opt.seqLength do
        entry[#entry+1] = string.format("%.5f" % loss_im[j])
      end
    end
    if not self.predIm and self.predFl then
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
  local lossSum_ps, lossSum_im, lossSum_fl, accSum_ps = {}, {}, {}, {}
  for i = 1, self.opt.seqLength do
    lossSum_ps[i] = 0.0
    lossSum_im[i] = 0.0
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

  self:setModelMode('evaluate')
  for i, sample in dataloader:run() do
    -- Get input and target
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local target_im = sample.target_im
    local target_fl = sample.target_fl

    -- Convert to CUDA
    input, target_ps, target_im, target_fl =
        self:convertCuda(input, target_ps, target_im, target_fl)

    -- Init output
    local loss_ps, loss_im, loss_fl = {}, {}, {}
    local acc_ps = {}

    -- Single nngraph model
    if torch.type(self.model) == 'nn.gModule' then
      -- Forward pass
      local output = self.model:forward(input)
      if self.predIm and not self.predFl then
        self.criterion:forward(self.model.output, {target_ps, target_im})
        for j = 1, #output[1] do
          if j <= self.seqlen then
            loss_ps[j] = self.criterion.criterions[1].criterions[j].output
            loss_im[j] = self.criterion.criterions[2].criterions[j].output
            acc_ps[j] = self:computeAccuracy(output[1][j]:contiguous(), target_ps[j])
          else
            loss_ps[j] = 0
            loss_im[j] = 0
            acc_ps[j] = 0/0
          end
          loss_fl[j] = 0/0
        end
      end
      if not self.predIm and self.predFl then
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
          loss_im[j] = 0/0
        end
      end
      if not self.predIm and not self.predFl then
        self.criterion:forward(self.model.output, target_ps)
        for j = 1, #output do
          if j <= self.seqlen then
            loss_ps[j] = self.criterion.criterions[j].output
            acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
          else
            loss_ps[j] = 0
            acc_ps[j] = 0/0
          end
          loss_im[j] = 0/0
          loss_fl[j] = 0/0
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
      local out_enc, out_rnn, inp_zero, inp_rnn
      for j = 1, self.seqlen do
        if j == 1 then
          out_enc = self.model['enc']:forward(input)
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
        self.criterion:forward(self.model['dec'].output, target_ps[j])

        loss_ps[j] = self.criterion.output
        acc_ps[j] = self:computeAccuracy(output[j]:contiguous(), target_ps[j])
        -- loss_im[j] = 0/0
        -- loss_fl[j] = 0/0
      end
      for j = self.seqlen+1, self.opt.seqLength do
        loss_ps[j] = 0
        acc_ps[j] = 0/0
        -- loss_im[j] = 0/0
        -- loss_fl[j] = 0/0
      end
      acc_ps = torch.Tensor(acc_ps)
    end

    -- Accumulate loss and acc
    if torch.all(acc_ps:sub(1,self.seqlen):eq(acc_ps:sub(1,self.seqlen))) then
      local batchSize = input:size(1)
      for j = 1, self.opt.seqLength do
        lossSum_ps[j] = lossSum_ps[j] + loss_ps[j]
        lossSum_im[j] = lossSum_im[j] + loss_im[j]
        lossSum_fl[j] = lossSum_fl[j] + loss_fl[j]
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
    lossSum_im[i] = lossSum_im[i] / N
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
  if self.predIm and not self.predFl then
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % lossSum_im[j])
    end
  end
  if not self.predIm and self.predFl then
    for j = 1, self.opt.seqLength do
      entry[#entry+1] = string.format("%.5f" % lossSum_fl[j])
    end
  end
  self.logger[split]:add(entry)
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local sidx = torch.LongTensor(dataloader:sizeSampled())
  local heatmaps, images, gtimages, flows, gtflows

  print("=> Generating predictions ...")
  xlua.progress(0, dataloader:sizeSampled())

  self:setModelMode('evaluate')
  for i, sample in dataloader:run({pred=true,samp=true}) do
    -- Get input and target
    local index = sample.index
    local input = sample.input[1]
    local target_ps = sample.target_ps
    local target_im = sample.target_im
    local target_fl = sample.target_fl

    -- Convert to CUDA
    input, target_ps, target_im, target_fl =
        self:convertCuda(input, target_ps, target_im, target_fl)

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

    -- Copy output
    if not heatmaps then
      heatmaps = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          target_ps[1]:size(2), self.opt.outputRes, self.opt.outputRes
      )
      if self.predIm and not self.predFl then
        images = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          3, self.opt.outputRes, self.opt.outputRes
        )
        gtimages = torch.FloatTensor(
          dataloader:sizeSampled(), self.opt.seqLength,
          3, self.opt.outputRes, self.opt.outputRes
        )
      end
      if not self.predIm and self.predFl then
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
    if self.predIm and not self.predFl then
      for j = 1, #output[1] do
        heatmaps[i][j]:copy(output[1][j][1])
        -- heatmaps[i][j]:copy(target_ps[2][j][1])
        -- Get image and target_im
        images[i][j]:copy(output[2][j][1])
        gtimages[i][j]:copy(target_im[j][1])
      end
    end
    if not self.predIm and self.predFl then
      for j = 1, #output[1] do
        heatmaps[i][j]:copy(output[1][j][1])
        -- heatmaps[i][j]:copy(target_ps[2][j][1])
        -- Get flow and target_fl
        flows[i][j]:copy(output[2][j][1])
        gtflows[i][j]:copy(target_fl[j][1])
      end
    end
    if not self.predIm and not self.predFl then
      for j = 1, #output do
        heatmaps[i][j]:copy(output[j][1])
        -- heatmaps[i][j]:copy(target_ps[j][1])
      end
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
  self:setModelMode('training')

  -- Sort heatmaps by index
  local sidx, i = torch.sort(sidx)
  heatmaps = heatmaps:index(1, i)
  if self.predIm and not self.predFl then
    images = images:index(1, i)
    gtimages = gtimages:index(1, i)
  end
  if not self.predIm and self.predFl then
    flows = flows:index(1, i)
    gtflows = gtflows:index(1, i)
  end

  -- Save final predictions
  local f = hdf5.open(self.opt.save .. '/preds_' .. split .. '.h5', 'w')
  -- local f = hdf5.open(self.opt.save .. '/gt_' .. split .. '.h5', 'w')
  f:write('heatmaps', heatmaps)
  f:close()

  -- Save image separately
  if self.predIm and not self.predFl then
    local f = hdf5.open(self.opt.save .. '/images_' .. split .. '.h5', 'w')
    f:write('images', images)
    f:write('gtimages', gtimages)
    f:close()
  end

  -- Save flow separately
  if not self.predIm and self.predFl then
    local f = hdf5.open(self.opt.save .. '/flows_' .. split .. '.h5', 'w')
    f:write('flows', flows)
    f:write('gtflows', gtflows)
    f:close()
  end
end

function Trainer:setSeqLenCritWeight(epoch)
  local seqlen
  -- Start len from 2 and no larger than data sequence length
  -- seqlen = math.ceil(epoch / self.opt.currInt) + 1
  seqlen = 2 ^ math.ceil(epoch / self.opt.currInt)
  seqlen = math.min(seqlen, self.opt.seqLength)
  -- Single nngraph model
  if torch.type(self.model) == 'nn.gModule' then
    if self.predIm or self.predFl then
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

function Trainer:convertCuda(input, target_ps, target_im, target_fl)
  input = input:cuda()
  for i = 1, #target_ps do
    target_ps[i] = target_ps[i]:cuda()
    target_im[i] = target_im[i]:cuda()
    target_fl[i] = target_fl[i]:cuda()
  end
  return input, target_ps, target_im, target_fl
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
