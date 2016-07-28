require 'lib/models/Residual'

M = {}

-- Input variables
local seqLength
local hiddenSize
local numLayers

local function hourglassEncoder(n, numIn, inp)
  local pool = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
  local low1 = Residual(numIn,256)(pool)
  local low2 = Residual(256,256)(low1)
  local low5 = Residual(256,256)(low2)
  if n > 1 then
    return hourglassEncoder(n-1,256,low5)
  else
    return low5
  end
end

local function hourglassDecoder(n, numOut, inp)
  local low6
  if n > 1 then
    low6 = hourglassDecoder(n-1,numOut,inp)
  else
    low6 = Residual(256,numOut)(inp)
  end
  local low7 = Residual(numOut,numOut)(low6)
  return nn.SpatialUpSamplingNearest(2)(low7)
end

local function lin(numIn,numOut,inp)
  -- Apply 1x1 convolution, no stride, no padding
  local l_ = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l_))
end

local function tieWeightBiasOneModule(module1, module2)
  if module2.modules ~= nil then
    assert(module1.modules)
    assert(#module1.modules == #module2.modules)
    for i = 1, #module1.modules do
      tieWeightBiasOneModule(module1.modules[i], module2.modules[i])
    end
  end
  if module2.weight ~= nil then
    assert(module1.weight)
    assert(module1.gradWeight)
    assert(module2.gradWeight)
    assert(torch.typename(module1) == torch.typename(module2))
    -- assert(module2.bias)
    module2.weight = module1.weight
    module2.gradWeight = module1.gradWeight
  end
  if module2.bias ~= nil then
    assert(module1.bias)
    assert(module1.gradBias)
    assert(module2.gradBias)
    assert(torch.typename(module1) == torch.typename(module2))
    -- assert(module2.weight)
    module2.bias = module1.bias
    module2.gradBias = module1.gradBias
  end
end

-- function fixWeightBiasOneModule(module)
--   if module.modules ~= nil then
--     for i = 1, #module.modules do
--       fixWeightBiasOneModule(module.modules[i])
--     end
--   end
--   if module.weight ~= nil then
--     assert(module.bias)
--     assert(module.accGradParameters)
--     -- TODO: remove accGradParametersBKP
--     module.accGradParametersBKP = module.accGradParameters
--     module.accGradParameters = function() end
--   end
-- end

function M.createModel(opt, outputDim)
  -- Set params
  seqLength = opt.seqLength
  hiddenSize = opt.hiddenSize
  numLayers = opt.numLayers
  outputRes = opt.outputRes

  -- Get input dim
  local Dataset = require('lib/datasets/' .. opt.dataset)
  local dataset = Dataset(opt, 'train')
  local outputDim = dataset.part:size(2)

  -- Input
  local inp = nn.Identity()()

  -- Initial processing of the image
  local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)
  local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
  local r1 = Residual(64,128)(cnv1)
  local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)
  local r4 = Residual(128,128)(pool)
  local r5 = Residual(128,128)(r4)
  local r6 = Residual(128,256)(r5)

  -- Encoder
  local enc = hourglassEncoder(4,256,r6)

  -- Merge into one mini-batch
  local x1_ = nn.Transpose({2,3},{3,4})(enc)
  local x1 = nn.View(-1,256)(x1_)

  -- LSTM
  local x2 = nn.View(-1,1,256)(x1)
  local x3 = nn.Padding(1,seqLength-1,1)(x2)
  local hid = cudnn.LSTM(256,hiddenSize,numLayers,true,0)(x3)
  local h1 = nn.Contiguous()(hid)

  -- Split from one mini-batch
  local h2_ = nn.View(-1,4,4,seqLength,256)(h1)
  local h2 = nn.Transpose({3,4},{2,3},{4,5},{3,4})(h2_)
  
  -- Merge output in batch dimension;
  local h3 = nn.View(-1,256,4,4)(h2)

  -- Decoder
  local dec = hourglassDecoder(4,512,h3)

  -- Linear layers to produce first set of predictions
  local l1 = lin(512,512,dec)
  local l2 = lin(512,512,l1)

  -- Output heatmaps
  local out1 = nnlib.SpatialConvolution(512,outputDim,1,1,1,1,0,0)(l2)

  -- Split output in batch dimension;
  local out2 = nn.View(-1,seqLength,outputDim,outputRes,outputRes)(out1)
  local out = nn.SplitTable(-4)(out2)

  -- Final model
  local model = nn.gModule({inp}, {out})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadHourglass(model, model_hg)
  local lstm_ind = 30
  local enc_eind = 25
  local dec_sind = 26
  local off_ind = 9

  -- Load encoder weight and bias
  for i = 1, enc_eind do
    local name = torch.typename(model.modules[i])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i])
  end

  -- Load decoder weight and bias
  for i = dec_sind, #model_hg.modules do
    local name = torch.typename(model.modules[i+off_ind])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i+off_ind])
  end

  -- -- Fix weight and bias
  -- for i = 1, enc_eind do
  --   fixWeightBiasOneModule(model.modules[i])
  -- end
  -- for i = dec_sind, #model_hg.modules do
  --   fixWeightBiasOneModule(model.modules[i+off_ind])
  -- end

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

function M.loadHourglassLSTM(model, model_hg)
  local lstm_ind = 30
  local lstm_ind_hg = 29
  local enc_eind = 25
  local dec_sind = 33
  local off_ind = 2

  -- Load encoder weight and bias
  for i = 1, enc_eind do
    local name = torch.typename(model.modules[i])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i])
  end

  -- Load decoder weight and bias
  for i = dec_sind, #model_hg.modules do
    local name = torch.typename(model.modules[i+off_ind])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i+off_ind])
  end

  -- Tie LSTM weight
  assert(torch.isTypeOf(model.modules[lstm_ind], cudnn.LSTM), 'lstm_ind error')
  assert(torch.isTypeOf(model_hg.modules[lstm_ind_hg], cudnn.LSTM), 
      'lstm_ind_hg error')
  tieWeightBiasOneModule(model_hg.modules[lstm_ind_hg], model.modules[lstm_ind])

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

return M
