require 'nngraph'
require 'cudnn'

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
    print('=> Creating model from file: lib/models/' .. opt.netType .. '.lua')
    local Model = require('lib/models/' .. opt.netType)
  
    -- Get output dim
    local Dataset = require('lib/datasets/' .. opt.dataset)
    local dataset = Dataset(opt, 'train')
    local outputDim = dataset.part:size(2)
  
    -- Create model
    model = Model.createModel(opt, outputDim)
  
    -- Load trained models
    if opt.hgModel ~= 'none' then
      assert(paths.filep(opt.hgModel),
          'initial hourglass model not found: ' .. opt.hgModel)
      local model_hg = torch.load(opt.hgModel)
      Model.loadHourglass(model, model_hg)
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

  -- Convert to CUDA
  model:cuda()
  criterion:cuda()

  return model, criterion
end

return M