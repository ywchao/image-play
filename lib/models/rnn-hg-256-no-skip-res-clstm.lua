require 'lib/models/Residual'

M = {}

-- Input variables
local seqLength
local hiddenSize
local numLayers
local dropout
local rememberStates = false

local function lstm(inp, inputSize)
  return cudnn.LSTM(inputSize,hiddenSize,numLayers,true,dropout,rememberStates)(inp)
end

local function encoder(n, f, inp)
  local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
  local low2 = Residual(f,f)(low1)
  local out
  if n > 1 then out = encoder(n-1,f,low2)
  else
    local low3 = Residual(f,f)(low2)

    out = {}
    table.insert(out,1,low3)
  end
  return out
end

local function rnn(n, f, inp)
  if n > 1 then out = rnn(n-1,f,inp)
  else
    local low = lstm(inp[#inp],f)
    out = {}
    table.insert(out,1,low)
  end
  return out
end

local function decoder(n, f, inp)
  local low3
  if n > 1 then low3 = decoder(n-1,f,inp)
  else
    low3 = inp[#inp]
  end
  local low4 = Residual(f,f)(low3)
  return nn.SpatialUpSamplingNearest(2)(low4)
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

function M.createModelEnc()
  -- Input
  local inp = nn.Identity()()

  -- Initial processing of the image
  local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)
  local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
  local r1 = Residual(64,128)(cnv1)
  local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)
  local r4 = Residual(128,128)(pool)
  local r5 = Residual(128,256)(r4)

  -- Encoder
  local out = encoder(4,256,r5)

  -- Final model
  local model = nn.gModule({inp}, out)

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.createModelRNN(opt)
  -- Set params
  seqLength = opt.seqLength
  hiddenSize = opt.hiddenSize
  numLayers = opt.numLayers
  outputRes = opt.outputRes
  dropout = opt.dropout

  -- Input to RNN
  local inp_rnn = {}
  inp_rnn[1] = nn.Identity()()

  -- Merge into one mini-batch
  local mer = {}
  mer[1] = nn.Transpose({2,3},{3,4})(inp_rnn[1])
  mer[1] = nn.View(-1,1,256)(mer[1])

  -- RNN
  local hid = rnn(4,256,mer)

  -- Split from one mini-batch
  local spl = {}
  local n = 0
  spl[1] = nn.View(-1,2^(n+2),2^(n+2),hiddenSize)(hid[1])
  spl[1] = nn.Transpose({3,4},{2,3})(spl[1])

  -- Full input
  local inp = {}
  inp[1] = nn.Identity()()
  inp[2] = inp_rnn[1]

  -- Add residual to encoder output
  local out = {}
  out[1] = nn.CAddTable()({inp[1],spl[1]})

  -- Decoder model
  local model = nn.gModule(inp, out)

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.createModelDec(outputDim)
  -- Input
  local inp = {}
  inp[1] = nn.Identity()()

  -- Decoder
  local dec = decoder(4,256,inp)

  -- Linear layers to produce first set of predictions
  local ll = lin(256,256,dec)

  -- Output heatmaps
  local out = nnlib.SpatialConvolution(256,outputDim,1,1,1,1,0,0)(ll)

  -- Decoder model
  local model = nn.gModule(inp, {out})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadHourglass(model_enc, model_dec, model_hg)
  -- Encoder
  for i = 1, #model_enc.modules do
    local name_enc = torch.typename(model_enc.modules[i])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name_enc == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model_enc.modules[i])
  end

  -- Decoder
  local gap = 1
  for i = #model_enc.modules+1, #model_hg.modules do
    local c = i - #model_enc.modules + gap
    local name_dec = torch.typename(model_dec.modules[c])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name_dec == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model_dec.modules[c])
  end

  -- Zero the gradients; not sure if this is necessary
  model_enc:zeroGradParameters()
  model_dec:zeroGradParameters()
end

return M
