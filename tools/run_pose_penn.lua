
dofile('./pose-hg-demo/util.lua')
dofile('./pose-hg-demo/img.lua')
dofile('./common/util.lua')

-- set gpu
-- currently only support gpu mode
cutorch.setDevice(arg[1])

-- set expID
if arg[2] then
  expID = arg[2]
  m = torch.load('./pose-hg-train/exp/mpii/' .. expID .. '/final_model.t7')
else
  expID = 'umich-stacked-hourglass'
  m = torch.load('./data/umich-stacked-hourglass/umich-stacked-hourglass.t7')
end

-- set paths
frame_root = './data/Penn_Action_cropped/frames/'
label_root = './data/Penn_Action_cropped/labels/'
cache_root = './caches/pose_penn_' .. expID .. '/'
vis_root = './caches/pose_penn_' .. expID .. '_vis/'

-- get image seq list
list_seq = dir(label_root, '.mat')
num_seq = table.getn(list_seq)

-- display a convenient progress bar
xlua.progress(0, num_seq)

for i = 1, num_seq do
  -- print(string.format('%04d/%04d',i,num_seq))
  -- read frames in sequence
  local name_seq = paths.basename(list_seq[i], '.mat')
  local fr_dir = frame_root .. name_seq .. '/'
  local list_fr = dir(fr_dir, '.jpg')
  local num_fr = table.getn(list_fr)
  -- set dir
  local cache_dir = cache_root .. name_seq .. '/'
  local vis_dir = vis_root .. name_seq .. '/'
  makedir(cache_dir)
  makedir(vis_dir)
  for j = 1, num_fr do
    -- skip if cache file exists
    local cache_file = cache_dir .. paths.basename(list_fr[j], '.jpg') .. '.h5'
    if paths.filep(cache_file) then
      goto continue
    end
    -- read image
    local file_im = fr_dir .. list_fr[j]
    local im = image.load(file_im)
    -- transform image
    local center = torch.Tensor({im:size()[3]/2,im:size()[2]/2})
    local scale = torch.max(torch.LongTensor(im:size()):sub(2,3)) / 200
    local inp = crop(im, center, scale, 0, 256)
    -- get network output
    local out = m:forward(inp:view(1,3,256,256):cuda())
    cutorch.synchronize()
    -- always use the last output
    local hm
    if type(out) == 'table' then
        hm = out[table.getn(out)][1]:float()
    else
        hm = out[1]:float()
    end
    hm[hm:lt(0)] = 0
    -- get predictions (hm and img refer to the coordinate space)
     preds_hm, preds_img = getPreds(hm, center, scale)
    -- save to file
    f = hdf5.open(cache_file, 'w')
    f:write('hm', hm)
    f:write('preds_hm', preds_hm)
    f:write('preds_img', preds_img)
    f:close()
    -- skip if vis file exists
    local vis_file = vis_dir .. list_fr[j]
    if paths.filep(vis_file) then
      goto continue
    end
    -- display and save the result
    preds_hm:mul(4)
    local dispImg = drawOutput(inp, hm, preds_hm[1])
    image.save(vis_file, dispImg)
    -- continue
    ::continue::
  end
  xlua.progress(i, num_seq)
end

