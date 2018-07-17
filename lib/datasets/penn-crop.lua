require 'hdf5'
require 'image'

local util = require 'common/util'
local img = require 'lib/util/img'
local geometry = require 'lib/util/geometry'

local M = {}
local PennCropDataset = torch.class('image-play.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  -- Load annotation
  annot_file = paths.concat(opt.data, split .. '.h5')
  self.ind2sub = hdf5.open(annot_file,'r'):read('ind2sub'):all()
  self.part = hdf5.open(annot_file,'r'):read('part'):all()
  -- Preprocess annotation
  self.seqId, self.nFrame = unpack(self:_preproAnno())
  -- Get phase number and LSTM sequence length
  self.nPhase = opt.nPhase
  self.seqLength = opt.seqLength
  -- Get input and output resolution
  self.inputRes = opt.inputRes
  self.outputRes = opt.outputRes
end

-- Get seq id and num of frames
function PennCropDataset:_preproAnno()
  local seqId = util.unique(self.ind2sub[{{},1}])
  local nFrame = {}
  for i = 1, seqId:numel() do
    local n = self.ind2sub[{{},1}]:eq(seqId[i]):sum()
    nFrame[i] = n
  end
  nFrame = torch.Tensor(nFrame)
  return {seqId, nFrame}
end

-- Get sequence in global index (ind2sub)
function PennCropDataset:_getSeq(i)
  local id = self.ind2sub[i][1]
  -- Get frame index
  local ii = self.seqId:eq(id)
  local nFrame = self.nFrame[ii][1]
  local ind = torch.linspace(i, i+nFrame-1, self.nPhase)
  local ind = torch.round(ind)
  assert(ind:numel() == self.nPhase)
  local ind = ind[{{1, self.seqLength}}]
  -- Replace overlength indices with the last index
  local rep_ind, rep_val
  -- ind2sub
  rep_ind = ind:gt(self.ind2sub:size(1))
  rep_val = torch.max(ind[rep_ind:eq(0)])
  ind[rep_ind] = rep_val
  -- sequence id
  rep_ind = self.ind2sub:index(1,ind:long())[{{},1}]:ne(id)
  rep_val = torch.max(ind[rep_ind:eq(0)])
  ind[rep_ind] = rep_val
  return ind
end

-- Get image path
function PennCropDataset:_impath(idx)
  return string.format('%04d/%06d.jpg',self.ind2sub[idx][1],self.ind2sub[idx][2])
end

-- Load image
function PennCropDataset:_loadImage(idx)
  return image.load(paths.concat(self.dir,self:_impath(idx)))
end

-- Get center and scale
function PennCropDataset:_getCenterScale(im)
  assert(im:size():size() == 3)
  local w = im:size(3)
  local h = im:size(2)
  local x = (w+1)/2
  local y = (h+1)/2
  local scale = math.max(w,h)/200
  -- Small adjustment so cropping is less likely to take feet out
  y = y + scale * 15
  scale = scale * 1.25
  return {torch.Tensor({x,y}), scale}
end

-- Get dataset size
function PennCropDataset:size()
  return self.ind2sub:size(1)
end

function PennCropDataset:get(idx, train)
  local input, repos, trans, focal, hmap, proj = {}, {}, {}, {}, {}, {}
  local center, scale
  local gtpts = {}

  local seqIdx = self:_getSeq(idx)
  for i = 1, seqIdx:numel() do
    local sid = seqIdx[i]
    -- Load image
    local im = self:_loadImage(sid)
    -- Get center and scale (same for all frames)
    if i == 1 then
      center, scale = unpack(self:_getCenterScale(im))
    end
    -- Transform image
    local inp = img.crop(im, center, scale, 0, self.inputRes)
    -- Get projection
    local pts = self.part[sid]
    local pj = torch.zeros(pts:size())
    for j = 1, pts:size(1) do
      if pts[j][1] ~= 0 and pts[j][2] ~= 0 then
        pj[j] = img.transform(pts[j], center, scale, 0, self.outputRes, false, false)
      end
    end
    -- Generate heatmap
    local hm = torch.zeros(pts:size(1), self.outputRes, self.outputRes)
    for j = 1, pts:size(1) do
      if pts[j][1] ~= 0 and pts[j][2] ~= 0 then
        img.drawGaussian(hm[j], torch.round(pj[j]), 2)
      end
    end
    input[i] = inp
    repos[i] = torch.zeros(pts:size(1),3)
    trans[i] = torch.zeros(3)
    focal[i] = torch.zeros(1)
    hmap[i] = hm
    proj[i] = pj
    gtpts[i] = pts
  end
  -- Augment data
  if train then
    -- Color
    local m1 = torch.uniform(0.8, 1.2)
    local m2 = torch.uniform(0.8, 1.2)
    local m3 = torch.uniform(0.8, 1.2)
    for i = 1, #input do
      input[i][{1, {}, {}}]:mul(m1):clamp(0, 1)
      input[i][{2, {}, {}}]:mul(m2):clamp(0, 1)
      input[i][{3, {}, {}}]:mul(m3):clamp(0, 1)
    end
    -- Flip
    if torch.uniform() <= 0.5 then
      for i = 1, #input do
        input[i] = img.flip(input[i])
        hmap[i] = img.flip(img.shuffleLR(hmap[i], 'penn-crop'))
        proj[i] = geometry.shuffleLR(proj[i],'penn-crop')
        local ind = proj[i]:eq(0)
        proj[i][{{},1}] = self.outputRes - proj[i][{{},1}] + 1
        proj[i][ind] = 0
      end
    end
  end
  return {
    input = input,
    repos = repos,
    trans = trans,
    focal = focal,
    hmap = hmap,
    proj = proj,
    gtpts = gtpts,
    center = center,
    scale = scale,
  }
end

function PennCropDataset:getSeqFrId(idx)
  return self.ind2sub[idx][1], self.ind2sub[idx][2]
end

-- Get sampled indices; for prediction and visualization
function PennCropDataset:getSampledIdx()
  local sidx
  local scnt = 0
  -- sidx should not depend on the input seqLength
  local seqLength_ = self.seqLength
  self.seqLength = 16
  for i = 1, self.ind2sub:size(1) do
    if self.ind2sub[i][2] == 1 then
      scnt = scnt + 1
      -- Subsample videos (1/10) for training set only
      if (self.split == 'train' or self.split == 'test') and scnt % 10 ~= 1 then
        goto continue
      end
      local seqIdx = self:_getSeq(i)
      if not sidx then
        sidx = seqIdx
      else
        sidx = torch.cat(sidx, seqIdx, 1)
      end
    end
    ::continue::
  end
  self.seqLength = seqLength_
  return sidx
end

return M.PennCropDataset