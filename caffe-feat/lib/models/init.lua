local M = {}

function M.setup(opt)
  -- Get model
  print('=> Creating model from file: lib/models/' .. opt.netType .. '.lua')
  local Model = require('lib/models/' .. opt.netType)

  -- Create model
  local model = Model.createModel(opt)

  -- Load pre-trained model
  assert(paths.filep(opt.prePrototxt),
      'Caffe model prototxt not found: ' .. opt.prePrototxt)
  assert(paths.filep(opt.preBinary),
      'Caffe model binary not found: ' .. opt.preBinary)
  Model.loadPretrained(model, opt.prePrototxt, opt.preBinary)

  -- Convert to CUDA
  model:cuda()

  return model
end

return M