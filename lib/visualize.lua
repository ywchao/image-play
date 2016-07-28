require 'image'
require 'common/util'
require 'lib/util/img'
require 'lib/util/eval'

M = {}

local function drawSkeleton(input, hms, coords, dataset)
  local im = input:clone()

  local pairRef, partNames, partColor, partColor, actThresh
  if dataset == 'penn-crop' then
    pairRef = {
        {2,1},{3,1},{4,2},{5,3},{6,4},{7,5},
        {8,2},{9,3},{10,8},{11,9},{12,10},{13,11}
    }
    partNames = {'Head','RSho','LSho','RElb','LElb','RWri','LWri',
                       'RHip','LHip','RKne','LKne','RAnk','LAnk'}
    partColor = {0,0,0,3,4,3,4,0,0,1,2,1,2}
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
  return im
end

local function drawOutput(input, hms, coords, dataset)
  local im = drawSkeleton(input, hms, coords, dataset)

  local colorHms = {}
  local inp64 = image.scale(input,64):mul(.3)
  for i = 1, hms:size(1) do
      colorHms[i] = colorHM(hms[i])
      colorHms[i]:mul(.7):add(inp64)
  end
  local totalHm = compileImages(colorHms, 4, 4, 64)
  im = compileImages({im,totalHm}, 1, 2, 256)
  im = image.scale(im,756)
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

  -- Load final predictions
  f = hdf5.open(opt.save .. '/preds_' .. split .. '.h5', 'r')
  heatmaps = f:read('heatmaps'):all()
  assert(heatmaps:size(1) == loaders[split]:sizeSampled())
  assert(heatmaps:size(2) == opt.seqLength)

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

      -- Get heatmap
      local idx = find(sidx, index[1])
      assert(idx:numel() == 1, 'index not found')
      local hm = heatmaps[idx[1]][j]:clone()
      hm[hm:lt(0)] = 0

      -- Get predictions
      local preds = getPreds(hm:view(1, hm:size(1), hm:size(2), hm:size(3)))
      preds = preds[1]

      -- Display and save the result
      preds:mul(4)
      local dispImg = drawOutput(inp:double(), hm, preds, opt.dataset)

      -- Save output
      image.save(vis_file, dispImg)

      -- Continue
      ::continue::
    end

    xlua.progress(i, dataloader:sizeSampled())
  end
end

return M
