require 'lib/models/Residual'

M = {}

-- Input variables
local seqLength
local hiddenSize
local numLayers

-- Variables below are hard-coded and should be adjusted based on the
-- network config
-- local lstm_ind = 29
-- local gap_enc = 3
-- local gap_dec = 3
local lstm_ind = 30
local gap_enc = 4
local gap_dec = 4

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

local function getLSTMLayerIndex(modules)
  local ind = {}
  for k, v in pairs(modules) do
    if torch.isTypeOf(v,cudnn.LSTM) then
      table.insert(ind, k)
    end
  end
  assert(#ind == 1)
  return ind[1]
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
  -- Tie parameters for BN layer
  if module2.running_mean ~= nil then
    assert(module1.running_mean)
    assert(module1.running_var)
    assert(module2.running_var)
    assert(torch.typename(module1) == torch.typename(module2))
    module2.running_mean = module1.running_mean
    module2.running_var = module1.running_var
  end
end

-- local function tieWeightBias(model, seqLength)
--   -- Get LSTM layer index and interval for replicated unit interval
--   assert(getLSTMLayerIndex(model.modules) == lstm_ind, 'lstm_ind error')
--   local int = (#model.modules - lstm_ind)/seqLength
--   -- Tie weight and bias for each module between reference and replicated units
--   for i = 1, int do
--     local f_ind = lstm_ind + i
--     local f_name = torch.typename(model.modules[f_ind])
--     for j = 2, seqLength do
--       local p_ind = lstm_ind + i + (j-1)*int
--       local p_name = torch.typename(model.modules[p_ind])
--       assert(f_name == p_name, 'weight tying error: class name mismatch')
--       tieWeightBiasOneModule(model.modules[f_ind], model.modules[p_ind])
--     end
--   end
--   -- collectgarbage()
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

  -- Max-pool the encoder output to get the input to LSTM
  -- local x1 = nnlib.SpatialMaxPooling(4,4,1,1)(enc)
  -- Fully connected layer to convert to vector
  local x1_ = nn.View(-1,256*4*4)(enc)
  local x1 = nn.Linear(256*4*4,256)(x1_)
  
  -- LSTM
  local x2 = nn.View(-1,1,256)(x1)
  local x3 = nn.Padding(1,seqLength-1,1)(x2)
  local hid = cudnn.LSTM(256,hiddenSize,numLayers,true,0)(x3)

  -- Weight and bias sharing approach 1: mini-batch merge and split
  -- Merge output in batch dimension; make hid contiguous first
  local h1_ = nn.Contiguous()(hid)
  local h1 = nn.View(-1,hiddenSize)(h1_)
  -- Fully connected and veiw layer to up-sample the hidden state
  local h2_ = nn.Linear(hiddenSize,256*4*4)(h1)
  local h2 = nn.View(-1,256,4,4)(h2_)
  -- Decoder
  local dec = hourglassDecoder(4,512,h2)
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

  -- -- Weight and bias sharing approach 2: sharing the same storage
  -- -- Generate output for each time step
  -- local out = {}
  -- for i = 1, seqLength do
  --   -- Extract the hidden state of one time step
  --   local h1 = nn.Narrow(2,i,1)(hid)
  --   -- Up-sample the hidden state to get the decoder input
  --   -- local h2 = nn.View(-1,hiddenSize,1,1)(h1)
  --   -- local h3 = nn.SpatialUpSamplingNearest(4)(h2)
  --   -- Fully connected and veiw layer to up-sample the hidden state
  --   local h2 = nn.View(-1,hiddenSize)(h1)
  --   local h3_ = nn.Linear(hiddenSize,256*4*4)(h2)
  --   local h3 = nn.View(-1,256,4,4)(h3_)
  --   -- Decoder
  --   local dec = hourglassDecoder(4,512,h3)
  --   -- Linear layers to produce first set of predictions
  --   local l1 = lin(512,512,dec)
  --   local l2 = lin(512,512,l1)
  --   -- Output heatmaps
  --   local oi = nnlib.SpatialConvolution(512,outputDim,1,1,1,1,0,0)(l2)
  --   table.insert(out, oi)
  -- end
  -- -- Final model
  -- local model = nn.gModule({inp}, out)
  -- -- Tie weight and bias
  -- tieWeightBias(model, seqLength)

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadHourglass(model, model_hg)
  -- Get LSTM layer index and interval for replicated unit interval
  assert(getLSTMLayerIndex(model.modules) == lstm_ind, 'lstm_ind error')
  -- local int = (#model.modules - lstm_ind)/seqLength

  -- Load encoder weight and bias
  for i = 1, lstm_ind-1-gap_enc do
    local name = torch.typename(model.modules[i])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i])
  end

  -- Load decoder weight and bias; only necessary for the reference unit
  local off_ind = gap_dec+1+gap_enc
  -- for i = lstm_ind+1+gap_dec, lstm_ind+int do
  --   local name = torch.typename(model.modules[i])
  --   local name_hg = torch.typename(model_hg.modules[i-off_ind])
  --   assert(name == name_hg, 'weight loading error: class name mismatch')
  --   tieWeightBiasOneModule(model_hg.modules[i-off_ind], model.modules[i])
  -- end
  for i = lstm_ind-gap_enc, #model_hg.modules do
    local name = torch.typename(model.modules[i+off_ind])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i+off_ind])
  end

  -- -- Tie weight and bias
  -- tieWeightBias(model, seqLength)

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

return M
