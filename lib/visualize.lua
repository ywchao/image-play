require 'image'
require 'common/util'
require 'lib/util/img'
require 'lib/util/eval'

M = {}

local function drawSkeleton(input, hms, coords, dataset)
  local im = input:clone()

  local pairRef, partNames, partColor, partColor, actThresh
  if dataset == 'penn-crop' then
    -- Interpolate thorax and pelvis
    local thms = hms
    local tcoords = coords
    coords = torch.Tensor(15,2)
    coords = coords:type(tcoords:type())
    coords:narrow(1,1,13):copy(tcoords)
    coords[14] = torch.mean(coords[{{2,3}}],1)
    coords[15] = torch.mean(coords[{{8,9}}],1)
    hms = torch.Tensor(15,thms:size(2),thms:size(3))
    hms = hms:type(thms:type())
    hms:narrow(1,1,13):copy(thms)
    -- local d142 = (coords[14]-coords[2]):div(4):round()
    -- local d143 = (coords[14]-coords[3]):div(4):round()
    -- local d158 = (coords[15]-coords[8]):div(4):round()
    -- local d159 = (coords[15]-coords[9]):div(4):round()
    -- hms[14] = (image.translate(hms[2],d142[1],d142[2]) +
    --            image.translate(hms[3],d143[1],d143[2])) / 2
    -- hms[15] = (image.translate(hms[8],d158[1],d158[2]) +
    --            image.translate(hms[9],d159[1],d159[2])) / 2
    hms[14]:fill(math.min(hms[2]:mean(),hms[3]:mean()))
    hms[15]:fill(math.min(hms[8]:mean(),hms[9]:mean()))

    pairRef = {
        {14,1},{2,14},{3,14},{4,2},{5,3},{6,4},{7,5},
        {15,14},{8,15},{9,15},{10,8},{11,9},{12,10},{13,11}
    }
    partNames = {'Head','RSho','LSho','RElb','LElb','RWri','LWri',
                        'RHip','LHip','RKne','LKne','RAnk','LAnak',
                        'Thrx','Pelv'}
    partColor = {0,3,4,3,4,3,4,1,2,1,2,1,2,0,0}
    actThresh = 0.002
  else
    error('unknown dataset')
  end

  -- Loop through adjacent joint pairings
  for i = 1, #pairRef do
    if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
      -- Set appropriate line color
      local color
      if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
      elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
      elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
      elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
      else color = {.7,0,.7} end

      -- Draw line
      im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
    end
  end
  return im, hms, coords
end

-- Draw predicted pose
local function drawOutput(input, hms, coords, dataset)
  local im, hms, coords = drawSkeleton(input, hms, coords, dataset)

  local colorHms = {}
  local inp64 = image.scale(input:double(),64):mul(.3)
  for i = 1, hms:size(1) do
      colorHms[i] = colorHM(hms[i])
      colorHms[i]:mul(.7):add(inp64)
  end
  local totalHm = compileImages(colorHms, 4, 4, 64)
  im = compileImages({im,totalHm}, 1, 2, 256)
  im = image.scale(im,756)
  return im
end

-- Draw predicted pose and predicted/gt image
local function drawOutputIm(input, hms, coords, prim, gtim, dataset)
  local gtim_ = image.scale(gtim,256)
  local prim_ = image.scale(prim,256)
  local im, hms, coords = drawSkeleton(prim_, hms, coords, dataset)

  local colorHms = {}
  local prim = prim:double():mul(0.3)
  for i = 1, hms:size(1) do
    colorHms[i] = colorHM(hms[i])
    colorHms[i]:mul(.7):add(prim)
  end
  local totalHm = compileImages(colorHms, 4, 4, 64)
  im = compileImages({input,gtim_,im,totalHm}, 1, 4, 256)
  im = image.scale(im,1512)
  return im
end

function M.run(loaders, split, opt, seqlen)
  seqlen = seqlen or opt.seqLength
  assert(seqlen <= opt.seqLength, 'visualizing sequence length error')

  local Dataset = require('lib/datasets/' .. opt.dataset)
  local dataset = Dataset(opt, split)
  local sidx = dataset:getSampledIdx()

  local dataloader = loaders[split]
  local vis_root = paths.concat(opt.save, 'preds_' .. split .. '_vis')
  -- local vis_root = paths.concat(opt.save, 'gt_' .. split .. '_vis')

  -- Load final predictions
  local f = hdf5.open(opt.save .. '/preds_' .. split .. '.h5', 'r')
  -- local f = hdf5.open(opt.save .. '/gt_' .. split .. '.h5', 'r')
  local heatmaps = f:read('heatmaps'):all()
  assert(heatmaps:size(1) == loaders[split]:sizeSampled())
  assert(heatmaps:size(2) == opt.seqLength)
  local images, gtimages
  if paths.filep(opt.save .. '/images_' .. split .. '.h5') then
    f = hdf5.open(opt.save .. '/images_' .. split .. '.h5', 'r')
    images = f:read('images'):all()
    gtimages = f:read('gtimages'):all()
    assert(images:size(1) == loaders[split]:sizeSampled())
    assert(images:size(2) == opt.seqLength)
    assert(gtimages:size(1) == loaders[split]:sizeSampled())
    assert(gtimages:size(2) == opt.seqLength)
  end

  print("=> Visualizing predictions ...")
  xlua.progress(0, dataloader:sizeSampled())

  for i, sample in dataloader:run({pred=true,samp=true}) do
    -- Get index and input
    local index = sample.index
    local input = sample.input

    assert(input[1]:size(1) == 1, 'batch size must be 1 with run({pred_true})')
    -- Get sid and fid; the current code might be exclusive to penn-crop;
    -- need customization for different datasets later
    local sid, fid = dataset:getSeqFrId(sidx[i])

    local vis_dir = paths.concat(vis_root, ('%04d'):format(sid))
    makedir(vis_dir)

    for j = 1, seqlen do
      local vis_file = paths.concat(vis_dir,
          ('%03d'):format(fid) .. '-' .. ('%02d'):format(j)) .. '.jpg'
      if paths.filep(vis_file) then
        goto continue
      end

      -- Use first frame as background
      local inp = input[1][1]
      -- local inp = input[j][1]

      -- Get heatmap and transformed image
      local idx = find(sidx, index[1])
      assert(idx:numel() == 1, 'index not found')
      local hm = heatmaps[idx[1]][j]:clone()
      hm[hm:lt(0)] = 0
      local prim, gtim
      if images ~= nil then
        prim = images[idx[1]][j]:clone()
        gtim = gtimages[idx[1]][j]:clone()
      end

      -- Get predictions
      local preds = getPreds(hm:view(1, hm:size(1), hm:size(2), hm:size(3)))
      preds = preds[1]

      -- Display and save the result
      preds:mul(4)
      local dispImg
      if images == nil then
        dispImg = drawOutput(inp, hm, preds, opt.dataset)
      else
        dispImg = drawOutputIm(inp, hm, preds, prim, gtim, opt.dataset)
      end

      -- Save output
      image.save(vis_file, dispImg)

      -- Continue
      ::continue::
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
end

return M
