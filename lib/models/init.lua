require 'nngraph'

local M = {}

function M.setup(opt, checkpoint)
  -- Get model
  local model
  local model_enc, model_rnn, model_dec
  if checkpoint then
    local modelPath = paths.concat(opt.save, checkpoint.modelFile)
    assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
    print('=> Resuming model from ' .. modelPath)
    model = torch.load(modelPath)
  else
    local Model = require('lib/models/' .. opt.netType)
  
    -- Get output dim
    local Dataset = require('lib/datasets/' .. opt.dataset)
    local dataset = Dataset(opt, 'train')
    local outputDim = dataset.part:size(2)
  
    -- Create model
    if Model.createModel ~= nil and
       Model.createModelEnc == nil and
       Model.createModelRNN == nil and
       Model.createModelDec == nil then
      model = Model.createModel(opt, outputDim)
    end
    if Model.createModel == nil and
       Model.createModelEnc ~= nil and
       Model.createModelRNN ~= nil and
       Model.createModelDec ~= nil then
      model = {}
      model['enc'] = Model.createModelEnc()
      model['rnn_one'] = Model.createModelRNN(opt)
      model['dec'] = Model.createModelDec(outputDim)
    end
  
    -- Load hourglass
    local model_hg
    if opt.hgModel ~= 'none' then
      assert(paths.filep(opt.hgModel),
          'initial hourglass model not found: ' .. opt.hgModel)
      model_hg = torch.load(opt.hgModel)
  
      local lstm_nodes = model_hg:findModules('cudnn.LSTM')
      if #lstm_nodes == 1 then
        -- Model.loadHourglassLSTM(model, model_hg)
        error('Not handling this case for now ... ')
      elseif #lstm_nodes == 0 then
        if torch.type(model) == 'nn.gModule' then
          Model.loadHourglass(model, model_hg)
        end
        if torch.type(model) == 'table' then
          Model.loadHourglass(model['enc'], model['dec'], model_hg)
        end
      else
        error('initial hourglass model error')
      end
    end
  end

  -- Create criterion
  -- Detect image prediction by checking the number of nn.SplitTable
  local criterion
  if torch.type(model) == 'nn.gModule' then
    if #model:findModules('nn.SplitTable') == 1 then
      -- Pose prediction only
      criterion = nn.ParallelCriterion()
      for i = 1, opt.seqLength do
        criterion:add(nn.MSECriterion())
      end
    end
    if #model:findModules('nn.SplitTable') == 2 then
      -- Pose and image prediction
      criterion = nn.ParallelCriterion()
      criterion:add(nn.ParallelCriterion())
      criterion:add(nn.ParallelCriterion())
      for i = 1, opt.seqLength do
        criterion.criterions[1]:add(nn.MSECriterion())
      end
      for i = 1, opt.seqLength do
        criterion.criterions[2]:add(nn.MSECriterion())
      end
    end
  end
  if torch.type(model) == 'table' then
    criterion = nn.MSECriterion()
    -- if #model.outnode.data.mapindex == 1 then
    --   -- Pose prediction only
    --   criterion = nn.MSECriterion()
    -- end
    -- if #model.outnode.data.mapindex == 2 then
    --   -- Pose and image prediction
    --   criterion = nn.ParallelCriterion()
    --   criterion:add(nn.MSECriterion())
    --   criterion:add(nn.MSECriterion())
    -- end
  end

  -- Convert to CUDA
  if torch.type(model) == 'nn.gModule' then
    model:cuda()
  end
  if torch.type(model) == 'table' then
    model['enc']:cuda()
    model['rnn_one']:cuda()
    model['dec']:cuda()
    -- Check if model is residual type
    if #model['rnn_one']:findModules('nn.CAddTable') == 0 then
      model['res'] = false
      model['vae'] = false
    else
      model['res'] = true
      -- Check if model contains VAE
      if model['rnn_one'].nInputs == #model['rnn_one'].outnode.data.mapindex * 2 then
        model['vae'] = false
      end
      if model['rnn_one'].nInputs == (#model['rnn_one'].outnode.data.mapindex-1) * (3/2) then
        model['vae'] = true
      end
    end
  end
  criterion:cuda()

  return model, criterion
end

return M
