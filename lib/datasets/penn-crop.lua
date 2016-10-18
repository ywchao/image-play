require 'hdf5'
require 'image'
require 'common/util'
require 'lib/util/img'

local M = {}
local PennCropDataset = torch.class('image-play.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  -- Load annotation
  annot_file = paths.concat(opt.data, split .. '.h5')
  self.ind2sub = hdf5.open(annot_file,'r'):read('ind2sub'):all()
  self.visible = hdf5.open(annot_file,'r'):read('visible'):all()
  self.part = hdf5.open(annot_file,'r'):read('part'):all()
  -- Preprocess annotation
  self.seqId, self.nFrame = unpack(self:_preproAnno())
  -- Get phase number and LSTM sequence length
  self.nPhase = opt.nPhase
  self.seqType = opt.seqType
  self.seqLength = opt.seqLength
  -- Get input and output resolution
  self.inputRes = opt.inputRes
  self.outputRes = opt.outputRes
end

-- Get seq id and num of frames
function PennCropDataset:_preproAnno()
  local seqId = unique(self.ind2sub[{{},1}])
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
  local ind
  if self.seqType == 'phase' then
    local ii = self.seqId:eq(id)
    local nFrame = self.nFrame[ii][1]
    ind = torch.linspace(i, i+nFrame-1, self.nPhase)
    ind = torch.round(ind)
    assert(ind:numel() == self.nPhase)
    ind = ind[{{1, self.seqLength}}]
  end
  if self.seqType == 'raw' then
    ind = torch.range(i,i+self.seqLength-1)
  end
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
function PennCropDataset:_imgpath(idx)
  return string.format('%04d/%06d.jpg',self.ind2sub[idx][1],self.ind2sub[idx][2])
end

-- Load image
function PennCropDataset:_loadImage(idx)
  return image.load(paths.concat(self.dir,self:_imgpath(idx)))
end

-- Get center and scale
function PennCropDataset:_getCenterScale(img)
  assert(img:size():size() == 3)
  local w = img:size(3)
  local h = img:size(2)
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
  local input = {}
  local label = {}
  local img, center, scale
  local inp, out

  local seqIdx = self:_getSeq(idx)
  for i = 1, seqIdx:numel() do
    sid = seqIdx[i]
    -- Load image
    img = self:_loadImage(sid)
    -- Get center and scale (same for all frames)
    if i == 1 then
      center, scale = unpack(self:_getCenterScale(img))
    end
    -- Transform image
    inp = crop(img, center, scale, 0, self.inputRes)
    -- Generate label
    local pts = self.part[sid]
    local vis = self.visible[sid]
    out = torch.zeros(pts:size(1), self.outputRes, self.outputRes)
    for j = 1, pts:size(1) do
      if vis[j] == 1 then
        drawGaussian(out[j], transform(pts[j], center, scale, 0, self.outputRes), 2)
      end
    end
    input[i] = inp
    label[i] = out
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
        input[i] = flip(input[i])
        label[i] = flip(shuffleLR(label[i], 'penn-crop'))
      end
    end
  end
  return {
    input = input,
    label = label,
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
  local seqType_ = self.seqType
  local seqLength_ = self.seqLength
  self.seqType = 'phase'
  self.seqLength = 16
  for i = 1, self.ind2sub:size(1) do
    if self.ind2sub[i][2] == 1 then
      scnt = scnt + 1
      -- Subsample videos (1/10) for training set only
      if self.split == 'train' and scnt % 10 ~= 1 then
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
  self.seqType = seqType_
  self.seqLength = seqLength_
  return sidx
end

return M.PennCropDataset