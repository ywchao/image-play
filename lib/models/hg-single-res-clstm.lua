require 'lib/models/Residual'

M = {}

-- Input variables
local seqLength
local hiddenSize
local numLayers

local function clstm(inp, inputSize, res)
  -- Replicate encoder output
  local rep = nn.Replicate(seqLength,2)(inp)

  -- Merge into one mini-batch
  local x1_ = nn.Transpose({2,3},{3,4})(inp)
  local x1 = nn.View(-1,inputSize)(x1_)

  -- LSTM
  local x2 = nn.View(-1,1,inputSize)(x1)
  local x3 = nn.Padding(1,seqLength-1,1)(x2)
  local hid = cudnn.LSTM(inputSize,hiddenSize,numLayers,true,0)(x3)
  local h1 = nn.Contiguous()(hid)

  -- Split from one mini-batch
  local h2_ = nn.View(-1,res,res,seqLength,hiddenSize)(h1)
  local h2 = nn.Transpose({3,4},{2,3},{4,5},{3,4})(h2_)
  
  -- Add residual to encoder output
  local add = nn.CAddTable()({rep,h2})

  -- Merge output in batch dimension;
  return nn.View(-1,hiddenSize,res,res)(add)
end

local function hourglass(n, numIn, numOut, inp)
  -- Upper branch
  local up1 = Residual(numIn,256)(inp)
  local up2 = Residual(256,256)(up1)
  local uph = clstm(up2,256,2^(n+2))
  local up4 = Residual(256,numOut)(uph)

  -- Lower branch
  local pool = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
  local low1 = Residual(numIn,256)(pool)
  local low2 = Residual(256,256)(low1)
  local low5 = Residual(256,256)(low2)
  local low6
  if n > 1 then
    low6 = hourglass(n-1,256,numOut,low5)
  else
    local lowh = clstm(low5,256,2^(n+1))
    low6 = Residual(256,numOut)(lowh)
  end
  local low7 = Residual(numOut,numOut)(low6)
  local up5 = nn.SpatialUpSamplingNearest(2)(low7)

  -- Bring two branches together
  return nn.CAddTable()({up4,up5})
end

local function lin(numIn,numOut,inp)
  -- Apply 1x1 convolution, no stride, no padding
  local l_ = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l_))
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

function M.createModel(opt, outputDim)
  -- Set params
  seqLength = opt.seqLength
  hiddenSize = opt.hiddenSize
  numLayers = opt.numLayers
  outputRes = opt.outputRes

  -- Input
  local inp = nn.Identity()()

  -- Initial processing of the image
  local cnv1_ = cudnn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)
  local cnv1 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
  local r1 = Residual(64,128)(cnv1)
  local pool = cudnn.SpatialMaxPooling(2,2,2,2)(r1)
  local r4 = Residual(128,128)(pool)
  local r5 = Residual(128,128)(r4)
  local r6 = Residual(128,256)(r5)

  -- Hourglass
  local hg = hourglass(4,256,512,r6)

  -- Linear layers to produce first set of predictions
  local l1 = lin(512,512,hg)
  local l2 = lin(512,512,l1)

  -- Output heatmaps
  local out1 = cudnn.SpatialConvolution(512,outputDim,1,1,1,1,0,0)(l2)

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
  -- Load weight and bias
  local clstm_gap = 11
  for i = 1, #model_hg.modules do
    -- Get corresponding module index in model
    local c = math.max(math.min(math.ceil((i-11)/7),4)*clstm_gap,0) + i
    if i > 9 + 7 * 4 then
      c = c + clstm_gap
    end
    -- Tie weight and bias
    local name = torch.typename(model.modules[c])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[c])
  end

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

return M
