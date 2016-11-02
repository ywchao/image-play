require 'hdf5'
require 'image'

local t = require 'lib/datasets/transforms'

local M = {}
local PennCropDataset = torch.class('caffe-feat.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  -- Load annotation
  annot_file = paths.concat(opt.data, split .. '.h5')
  self.ind2sub = hdf5.open(annot_file,'r'):read('ind2sub'):all()
  -- Get resize and crop size
  self.resizeRes = opt.resizeRes
  self.cropRes = opt.cropRes
  -- Get mean
  self.mean = opt.mean
end

-- Get image path
function PennCropDataset:_imgpath(idx)
  return string.format('%04d/%06d.jpg',self.ind2sub[idx][1],self.ind2sub[idx][2])
end

-- Load image
function PennCropDataset:_loadImage(idx)
  return image.load(paths.concat(self.dir,self:_imgpath(idx)), 3, 'byte')
end

function PennCropDataset:_preprocess(im, train, flip)
  local input = im
  input = image.scale(input, self.resizeRes, self.resizeRes)
  input = input:float()
  for i = 1, 3 do
    input[i]:csub(self.mean[i])
  end
  input = input:index(1,torch.LongTensor({3,2,1}))
  input = t.TenCrop(self.cropRes)(input)
  return input
end

-- Get dataset size
function PennCropDataset:size()
  return self.ind2sub:size(1)
end

function PennCropDataset:get(idx)
  -- Load image
  local im = self:_loadImage(idx)

  -- Preprocess input
  local input = self:_preprocess(im)

  return {
    input = input,
  }
end

function PennCropDataset:getSeqFrId(idx)
  return self.ind2sub[idx][1], self.ind2sub[idx][2]
end

return M.PennCropDataset