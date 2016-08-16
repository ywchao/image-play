require 'hdf5'
require 'image'
require 'common/util'
require 'lib/util/img'

local M = {}
local PennCropDataset = torch.class('image-play.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  self.opt = opt
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

-- Get phase sequence in global index (ind2sub)
function PennCropDataset:_getPhaseSeq(i)
  local id = self.ind2sub[i][1]
  local ii = self.seqId:eq(id)
  local nFrame = self.nFrame[ii][1]
  -- Get frame index
  local ind = torch.linspace(i, i+nFrame-1, self.nPhase)
  local ind = torch.round(ind)
  assert(ind:numel() == self.nPhase)
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
  -- Drop frames over LSTM sequence length
  if ind:size(1) ~= self.seqLength then
    ind = ind[{{1, self.seqLength}}]
  end
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
  return {torch.Tensor({x,y}), scale}
end

-- Get dataset size
function PennCropDataset:size()
  return self.ind2sub:size(1)
end

function PennCropDataset:get(idx)
  local input = {}
  -- local input
  local target = {}
  local img, center, scale
  local inp, out

  local phaseSeq = self:_getPhaseSeq(idx)
  for i = 1, phaseSeq:numel() do
    pidx = phaseSeq[i]
    -- Load image
    img = self:_loadImage(pidx)
    -- Get center and scale (same for all frames)
    if i == 1 then
      center, scale = unpack(self:_getCenterScale(img))
    end
    -- Transform image
    inp = crop(img, center, scale, 0, self.inputRes)
    -- Generate target
    local pts = self.part[pidx]
    local vis = self.visible[pidx]
    out = torch.zeros(pts:size(1), self.outputRes, self.outputRes)
    for j = 1, pts:size(1) do
      if vis[j] == 1 then
        drawGaussian(out[j], transform(torch.add(pts[j],1), center, scale, 0, self.outputRes), 2)
      end
    end
    input[i] = inp
    -- We only need the first frame for input
    -- if i == 1 then
    --   input = torch.Tensor(inp:size()):copy(inp)
    -- end
    target[i] = out
  end
  return {
    input = input,
    target = target,
  }
end

function PennCropDataset:getSeqFrId(idx)
  return self.ind2sub[idx][1], self.ind2sub[idx][2]
end

-- Get matched parts; for data augmentation
function PennCropDataset:getMatchedParts()
  return {{2,3},{4,5},{6,7},{8,9},{10,11},{12,13}}
end

-- Get sampled indices; for prediction and visualization
function PennCropDataset:getSampledIdx()
  local sidx
  local scnt = 0
  -- sidx should not depend on the input seqLength
  local tmp = self.seqLength
  self.seqLength = 16
  for i = 1, self.ind2sub:size(1) do
    if self.ind2sub[i][2] == 1 then
      scnt = scnt + 1
      -- Subsample videos (1/10) for training set only
      if self.split == 'train' and scnt % 10 ~= 1 then
        goto continue
      end
      local phaseSeq = self:_getPhaseSeq(i)
      if not sidx then
        sidx = phaseSeq
      else
        sidx = torch.cat(sidx, phaseSeq, 1)
      end
    end
    ::continue::
  end
  self.seqLength = tmp
  return sidx
end

return M.PennCropDataset
