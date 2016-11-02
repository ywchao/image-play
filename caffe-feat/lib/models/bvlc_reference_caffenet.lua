require 'nngraph'
require 'cudnn'
require 'loadcaffe'

local M = {}

local function tieWeightBiasOneModule(module1, module2)
  assert(module2.modules == nil)
  if module2.weight ~= nil then
    assert(module1.weight)
    assert(module1.gradWeight)
    assert(module2.gradWeight)
    assert(module2.weight:numel() == module1.weight:numel())
    assert(module2.gradWeight:numel() == module1.gradWeight:numel())
    assert(torch.typename(module1) == torch.typename(module2))
    module2.weight = module1.weight
    module2.gradWeight = module1.gradWeight
  end
  if module2.bias ~= nil then
    assert(module1.bias)
    assert(module1.gradBias)
    assert(module2.gradBias)
    assert(module2.bias:numel() == module1.bias:numel())
    assert(module2.gradBias:numel() == module1.gradBias:numel())
    assert(torch.typename(module1) == torch.typename(module2))
    module2.bias = module1.bias
    module2.gradBias = module1.gradBias
  end
end

function M.createModel(opt)
  -- Input
  local inp = nn.Identity()()

  local conv1 = cudnn.SpatialConvolution(3,96,11,11,4,4,0,0)(inp)
  local relu1 = cudnn.ReLU(true)(conv1)
  local pool1 = cudnn.SpatialMaxPooling(3,3,2,2)(relu1)
  local norm1 = cudnn.SpatialCrossMapLRN(5)(pool1)
  local conv2 = cudnn.SpatialConvolution(96,256,5,5,1,1,2,2,2)(norm1)
  local relu2 = cudnn.ReLU(true)(conv2)
  local pool2 = cudnn.SpatialMaxPooling(3,3,2,2)(relu2)
  local norm2 = cudnn.SpatialCrossMapLRN(5)(pool2)
  local conv3 = cudnn.SpatialConvolution(256,384,3,3,1,1,1,1)(norm2)
  local relu3 = cudnn.ReLU(true)(conv3)
  local conv4 = cudnn.SpatialConvolution(384,384,3,3,1,1,1,1,2)(relu3)
  local relu4 = cudnn.ReLU(true)(conv4)
  local conv5 = cudnn.SpatialConvolution(384,256,3,3,1,1,1,1,2)(relu4)
  local relu5 = cudnn.ReLU(true)(conv5)
  local pool5 = cudnn.SpatialMaxPooling(3,3,2,2)(relu5)

  local view5 = nn.View(-1):setNumInputDims(3)(pool5)
  local fc6 = nn.Linear(9216,4096)(view5)
  local relu6 = cudnn.ReLU(true)(fc6)
  local drop6 = nn.Dropout(0.5)(relu6)
  local fc7 = nn.Linear(4096,4096)(drop6)
  local relu7 = cudnn.ReLU(true)(fc7)
  local drop7 = nn.Dropout(0.5)(relu7)

  conv1.data.module.name = 'conv1'
  relu1.data.module.name = 'relu1'
  pool1.data.module.name = 'pool1'
  norm1.data.module.name = 'norm1'
  conv2.data.module.name = 'conv2'
  relu2.data.module.name = 'relu2'
  pool2.data.module.name = 'pool2'
  norm2.data.module.name = 'norm2'
  conv3.data.module.name = 'conv3'
  relu3.data.module.name = 'relu3'
  conv4.data.module.name = 'conv4'
  relu4.data.module.name = 'relu4'
  conv5.data.module.name = 'conv5'
  relu5.data.module.name = 'relu5'
  pool5.data.module.name = 'pool5'

  view5.data.module.name = 'torch_view'
  fc6.data.module.name = 'fc6'
  relu6.data.module.name = 'relu6'
  drop6.data.module.name = 'drop6'
  fc7.data.module.name = 'fc7'
  relu7.data.module.name = 'relu7'
  drop7.data.module.name = 'drop7'

  -- Output
  local out = drop7

  -- Final model
  local model = nn.gModule({inp}, {out})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadPretrained(model, prototxt_path, binary_path)
  local model_pre = loadcaffe.load(prototxt_path, binary_path, 'cudnn')

  -- Load weight and bias
  for _, v in ipairs(model_pre.modules) do
    model:apply(function(module)
      if module.name == v.name then
        assert(torch.typename(module) == torch.typename(v),
            'weight loading error: class name mismatch')
        tieWeightBiasOneModule(v, module)
      end
    end)
  end

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

return M