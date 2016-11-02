local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 caff_feat running script')
  cmd:text()
  cmd:text('Options:')
  cmd:text(' ------------ General options --------------------')
  cmd:option('-dataset',      'penn-crop', 'Options: penn-crop')
  cmd:option('-data',  './data/penn-crop', 'Path to dataset')
  cmd:option('-manualSeed',             3, 'Manually set RNG seed')
  cmd:option('-GPU',                    1, 'Default preferred GPU')
  cmd:option('-expDir',           './exp', 'Directory in which to save/log experiments')
  cmd:option('-expID',          'default', 'Experiment ID')
  cmd:text(' ------------ Data options -----------------------')
  cmd:option('-nThreads',               4, 'Number of data loading threads')
  cmd:text(' ------------ Model options ----------------------')
  cmd:option('-netType', 'bvlc_reference_caffenet', 'Options: ')
  cmd:option('-resizeRes',            256, 'Input image resolution')
  cmd:option('-cropRes',              227, 'Random cropping resolution')
  cmd:option('-prePrototxt', 'caffe/models/bvlc_reference_caffenet/deploy.prototxt', 'Pre-trained Caffe model prototxt')
  cmd:option('-preBinary', 'caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 'Pre-trained Caffe model binary')
  cmd:option('-mean',     {123, 117, 104}, 'Pixel mean values (RGB order)')
  cmd:text()

  local opt = cmd:parse(arg or {})

  opt.expDir = paths.concat(opt.expDir, opt.dataset)
  opt.save = paths.concat(opt.expDir, opt.expID)

  if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
    cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
  end

  return opt
end

return M