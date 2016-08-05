require 'nngraph'

local M = {}

function M.setup(opt, checkpoint)
  -- Get model
  local model
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
    model = Model.createModel(opt, outputDim)
  
    -- Load hourglass
    local model_hg
    if opt.hgModel ~= 'none' then
      assert(paths.filep(opt.hgModel),
          'initial hourglass model not found: ' .. opt.hgModel)
      model_hg = torch.load(opt.hgModel)
  
      local lstm_nodes = model_hg:findModules('cudnn.LSTM')
      if #lstm_nodes == 1 then
        Model.loadHourglassLSTM(model, model_hg)
      elseif #lstm_nodes == 0 then
        Model.loadHourglass(model, model_hg)
      else
        error('initial hourglass model error')
      end
    end
  end

  -- Create criterion
  -- Detect image prediction by checking the number of nn.SplitTable
  local criterion
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

  -- Convert to CUDA
  -- TODO: handle CPU case
  model:cuda()
  criterion:cuda()

  return model, criterion
end

return M
