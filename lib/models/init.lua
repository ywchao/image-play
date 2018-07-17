require 'nngraph'
require 'cudnn'

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
    if opt.hgModel ~= 'none' then
      assert(paths.filep(opt.hgModel),
          'initial hourglass model not found: ' .. opt.hgModel)
      local model_hg = torch.load(opt.hgModel)
  
      if torch.type(model) == 'nn.gModule' then
        Model.loadHourglass(model, model_hg)
      end
      if torch.type(model) == 'table' then
        Model.loadHourglass(model['enc'], model['dec'], model_hg)
      end
    end
    if opt.s3Model ~= 'none' then
      assert(paths.filep(opt.s3Model),
          'initial skel3dnet model not found: ' .. opt.s3Model)
      local model_s3 = torch.load(opt.s3Model)
      Model.loadSkel3DNet(model, model_s3)
    end
    if opt.hgs3Model ~= 'none' then
      assert(opt.hgModel == 'none' and opt.s3Model == 'none')
      assert(paths.filep(opt.hgs3Model),
          'initial hg/s3 combined model not found: ' .. opt.hgs3Model)
      local model_hgs3 = torch.load(opt.hgs3Model)
      Model.loadHGS3(model, model_hgs3)
    end
  end

  -- Create criterion
  local criterion
  local nOutput = #model.outnode.children
  if torch.type(model) == 'nn.gModule' then
    assert(nOutput == 1 or nOutput == 5)
    criterion = nn.ParallelCriterion()
    if nOutput == 1 then
      for i = 1, opt.seqLength do
        criterion:add(nn.MSECriterion())
      end
    end
    if nOutput == 5 then
      for i = 1, nOutput do
        criterion:add(nn.ParallelCriterion())
        for j = 1, opt.seqLength do
          criterion.criterions[i]:add(nn.MSECriterion())
        end
      end
    end
  end
  if torch.type(model) == 'table' then
    criterion = nn.MSECriterion()
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
    else
      model['res'] = true
    end
  end
  criterion:cuda()

  return model, criterion
end

return M