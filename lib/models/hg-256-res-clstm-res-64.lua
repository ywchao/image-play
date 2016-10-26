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

local function hourglass(n, f, inp)
  -- Upper branch
  local up1 = Residual(f,f)(inp)
  local up1 = clstm(up1,f,2^(n+2))

  -- Lower branch
  local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
  local low2 = Residual(f,f)(low1)
  local low3

  if n > 1 then low3 = hourglass(n-1,f,low2)
  else
    low3 = Residual(f,f)(low2)
    low3 = clstm(low3,f,2^(n+1))
  end
  local low4 = Residual(f,f)(low3)
  local up2 = nn.SpatialUpSamplingNearest(2)(low4)

  -- Bring two branches together
  return nn.CAddTable()({up1,up2})
end

local function lin(numIn, numOut, inp)
  -- Apply 1x1 convolution, no stride, no padding
  local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

local function encoder(n, f, inp)
  -- Lower branch
  local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
  local low2 = Residual(f,f)(low1)
  local low3
  if n > 1 then low3 = encoder(n-1,f,low2)
  else
    low3 = Residual(f,f)(low2)
  end
  return low3
end

local function skel3dnet(inp, numPt, outputRes)
  -- Initial processing of the heatmaps
  local hm1 = cudnn.SpatialConvolution(numPt,64,1,1,1,1,0,0)(inp)
  local hm2 = cudnn.SpatialBatchNormalization(64)(hm1)
  local hm3 = cudnn.ReLU(true)(hm2)

  local cntr = encoder(4,64,hm3)
  local view = nn.View(-1):setNumInputDims(3)(cntr)

  local dfc = (outputRes/2^4)^2*64

  -- Relative joint position
  local fc1 = nn.Linear(dfc,dfc/4)(view)
  local relu1 = cudnn.ReLU(true)(fc1)
  local repos = nn.Linear(dfc/4,numPt*3)(relu1)
  local repos = nn.View(-1,numPt,3)(repos)

  -- Translation of skeleton center
  local fc2 = nn.Linear(dfc,dfc/4)(view)
  local relu2 = cudnn.ReLU(true)(fc2)
  local trans = nn.Linear(dfc/4,3)(relu2)
  local txy = nn.Narrow(2,1,2)(trans)
  local td = nn.Narrow(2,3,1)(trans)
  local td = nn.AddConstant(2500)(td)
  local trans = nn.JoinTable(2)({txy,td})

  -- Focal length
  local fc3 = nn.Linear(dfc,dfc/4)(view)
  local relu3 = cudnn.ReLU(true)(fc3)
  local focal = nn.Linear(dfc/4,1)(relu3)
  local focal = nn.AddConstant(73.6)(focal)

  -- 3D points in camera coordinates
  local rept = nn.Replicate(numPt,2)(trans)
  local p3d = nn.CAddTable()({repos,rept})

  -- Projection
  local xy = nn.Narrow(3,1,2)(p3d)
  local d = nn.Select(3,3)(p3d)
  local repd = nn.Replicate(2,3)(d)
  local proj = nn.CDivTable()({xy,repd})
  local f = nn.Select(2,1)(focal)
  local repf = nn.Replicate(numPt,2)(f)
  local repf = nn.Replicate(2,3)(repf)
  local proj = nn.CMulTable()({proj,repf})
  local proj = nn.AddConstant(outputRes/2)(proj)

  return proj, repos, trans, focal
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
  if module2.running_mean ~= nil then
    assert(module1.running_mean)
    assert(module1.running_var)
    assert(module2.running_var)
    assert(torch.typename(module1) == torch.typename(module2))
    assert(module2.running_mean:numel() == module1.running_mean:numel())
    assert(module2.running_var:numel() == module1.running_var:numel())
    module2.running_mean = module1.running_mean
    module2.running_var = module1.running_var
  end
end

local function fixWeightBiasOneModule(module)
  if module.modules ~= nil then
    for i = 1, #module.modules do
      fixWeightBiasOneModule(module.modules[i])
    end
  end
  if module.weight ~= nil then
    assert(module.bias)
    assert(module.accGradParameters)
    module.accGradParameters = function() end
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
  local in1 = cudnn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)
  local in2 = nn.SpatialBatchNormalization(64)(in1)
  local in3 = cudnn.ReLU(true)(in2)
  local in4 = Residual(64,128)(in3)
  local in5 = cudnn.SpatialMaxPooling(2,2,2,2)(in4)
  local in6 = Residual(128,128)(in5)
  local in7 = Residual(128,256)(in6)

  -- Hourglass
  local hg = hourglass(4,256,in7)

  -- Linear layers to produce first set of predictions
  local ll = lin(256,256,hg)

  -- Predicted heatmaps
  local hmap = cudnn.SpatialConvolution(256,outputDim,1,1,1,1,0,0)(ll)

  -- 3D skeleton net
  local proj, repos, trans, focal = skel3dnet(hmap, outputDim, outputRes)

  -- Split in batch dimension
  local view1 = nn.View(-1,seqLength,outputDim,outputRes,outputRes)(hmap)
  local hmap = nn.SplitTable(-4)(view1)
  local view2 = nn.View(-1,seqLength,outputDim,3)(repos)
  local repos = nn.SplitTable(-3)(view2)
  local view3 = nn.View(-1,seqLength,3)(trans)
  local trans = nn.SplitTable(-2)(view3)
  local view4 = nn.View(-1,seqLength,1)(focal)
  local focal = nn.SplitTable(-2)(view4)
  local view5 = nn.View(-1,seqLength,outputDim,2)(proj)
  local proj = nn.SplitTable(-3)(view5)

  -- Final model
  local model = nn.gModule({inp}, {hmap, repos, trans, focal, proj})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadHourglass(model, model_hg)
  -- Load weight and bias
  local clstm_gap = 11
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

function M.loadSkel3DNet(model, model_s3)
  -- Load weight and bias
  local offset = 93
  for i = 2, #model_s3.modules do
    local c = i + offset
    if i >= 19 then c = c + 2 end
    if i >= 26 then c = c + 2 end
    local name = torch.typename(model.modules[c])
    local name_s3 = torch.typename(model_s3.modules[i])
    assert(name == name_s3, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_s3.modules[i], model.modules[c])
    -- Fix parameters for now
    fixWeightBiasOneModule(model.modules[c])
  end

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

function M.loadHGS3(model, model_hgs3)
  -- Load weight and bias
  local clstm_gap = 11
  for i = 1, #model_hgs3.modules do
    -- Get corresponding module index in model
    local c = math.max(math.min(math.ceil((i-9)/3),4)*clstm_gap,0) + i
    if i > 8 + 3 * 4 + 1 then
      c = c + clstm_gap
    end
    if i >= 38 then c = c + 2 end
    if i >= 55 then c = c + 2 end
    if i >= 62 then c = c + 2 end
    if i >= 66 then c = c + 2 end
    -- Tie weight and bias
    local name = torch.typename(model.modules[c])
    local name_hgs3 = torch.typename(model_hgs3.modules[i])
    assert(name == name_hgs3, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hgs3.modules[i], model.modules[c])
    -- Fix s3 parameters for now
    if i >= 95 then
      fixWeightBiasOneModule(model.modules[c])
    end
  end
end

return M