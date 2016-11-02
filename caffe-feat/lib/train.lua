local matio = require 'matio'
local util = require 'common/util'

local M = {}
local Trainer = torch.class('caffe-feat.Trainer', M)

function Trainer:__init(model, opt)
  self.model = model
  self.opt = opt
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local size = dataloader:size()
  local inds = torch.IntTensor(size)

  -- local Dataset = require('lib/datasets/' .. self.opt.dataset)
  -- local dataset = Dataset(self.opt, split)
  local feat

  print("=> Generating predictions ...")
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run() do
    -- Get input and convert to CUDA
    local index = sample.index
    local input = sample.input:cuda()

    -- Forward pass
    local output = self.model:forward(input)
    output = output:float()
    output = output:view(1,10,output:size(2)):sum(2):squeeze(2)

    -- Save output feature
    assert(input:size(1) == 10, 'batch size must be 10')
    -- if self.opt.dataset == 'penn-crop' then
    --   local sid, fid = dataset:getSeqFrId(index[1])
    --   local feat_path = paths.concat(self.opt.save,'feat_' .. split, string.format('%04d' % sid))
    --   local feat_file = paths.concat(feat_path, string.format("%06d.mat" % fid))
    --   util.makedir(feat_path)
    --   if not paths.filep(feat_file) then
    --     matio.save(feat_file, {feat = output[1]})
    --   end
    -- end
    if not feat then
      feat = torch.FloatTensor(size,output[1]:numel())
    end
    feat[index[1]] = output[1]

    xlua.progress(i, size)
  end
  self.model:training()

  -- save output feature
  local feat_file = paths.concat(self.opt.save,'feat_' .. split .. '.mat')
  if not paths.filep(feat_file) then
    matio.save(feat_file, {feat = feat})
  end
end

return M.Trainer