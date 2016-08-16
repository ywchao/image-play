require 'stn'
require 'lib/models/Residual'

M = {}

-- Input variables
local seqLength
local hiddenSize
local numLayers

local function clstm(inp, inputSize, res)
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
  
  -- Merge output in batch dimension;
  return nn.View(-1,hiddenSize,res,res)(h2)
end

local function hourglass(n, f, inp)
  -- Upper branch
  local up1 = Residual(f,f)(inp)
  local up2 = clstm(up1,f,2^(n+2))

  -- Lower branch
  local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
  local low2 = Residual(f,f)(low1)
  local low3p, low3f

  if n > 1 then low3p, low3f = hourglass(n-1,f,low2)
  else
    low3p = Residual(f,f)(low2)
    low3p = clstm(low3p,f,2^(n+1))
    low3f = low3p
  end
  local low4p = Residual(f,f)(low3p)
  local up3p = nn.SpatialUpSamplingNearest(2)(low4p)
  local low4f = Residual(f,f)(low3f)
  local up3f = nn.SpatialUpSamplingNearest(2)(low4f)

  -- Bring two branches together
  return nn.CAddTable()({up2,up3p}), nn.CAddTable()({up2,up3f})
end

local function lin(numIn,numOut,inp)
  -- Apply 1x1 convolution, no stride, no padding
  local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
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
  local r5 = Residual(128,256)(r4)

  -- Hourglass
  local hgp, hgf = hourglass(4,256,r5)

  -- Linear layers to produce first set of predictions
  local llp = lin(256,256,hgp)
  local llf = lin(256,256,hgf)

  -- Output heatmaps and flow
  local hmp = nnlib.SpatialConvolution(256,outputDim,1,1,1,1,0,0)(llp)
  local flo = nnlib.SpatialConvolution(256,2,1,1,1,1,0,0)(llf)

  -- Split output in batch dimension;
  local outp_ = nn.View(-1,seqLength,outputDim,outputRes,outputRes)(hmp)
  local outp = nn.SplitTable(-4)(outp_)
  local outf_ = nn.View(-1,seqLength,2,outputRes,outputRes)(flo)
  local outf = nn.SplitTable(-4)(outf_)

  -- Final model
  local model = nn.gModule({inp}, {outp, outf})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadHourglass(model, model_hg)
  -- Load weight and bias
  local clstm_gap = 9
  for i = 1, #model_hg.modules do
    -- Get corresponding module index in model
    local c = math.max(math.min(math.ceil((i-9)/3),4)*clstm_gap,0) + i
    if i > 8 + 3 * 4 + 1 then
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
