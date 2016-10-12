local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 image-play training script')
  cmd:text()
  cmd:text('Options:')
  -- ------------ General options --------------------
  cmd:text(' ------------ General options --------------------')
  cmd:option('-data',  './data/Penn_Action_cropped', 'Path to dataset')
  cmd:option('-dataset',   'penn-crop', 'Options: penn-crop')
  cmd:option('-manualSeed',          3, 'Manually set RNG seed')
  cmd:option('-GPU',                 1, 'Default preferred GPU')
  cmd:option('-expDir',        './exp', 'Directory in which to save/log experiments')
  cmd:option('-expID',       'default', 'Experiment ID')
  cmd:text(' ------------ Data options -----------------------')
  cmd:option('-nThreads',            1, 'Number of data loading threads')
  cmd:text(' ------------ Training options -------------------')
  cmd:option('-nEpochs',             0, 'Number of total epochs to run')
  cmd:option('-batchSize',           4, 'Training mini-batch size (1 = pure stochastic)')
  cmd:option('-seqType',        'phase','Options: phase | raw')
  cmd:option('-nPhase',             16, 'Number of phase for each action')
  cmd:option('-currInt',             1, 'Number of epochs before sequence length increment')
  cmd:option('-testInt',          5000, 'Test interval during training')
  cmd:text(' ------------ Checkpointing options --------------')
  cmd:option('-resume',          false,'Resume from the latest checkpoint in this directory')
  cmd:text(' ------------ Optimization  options --------------')
  cmd:option('-LR',             2.5e-4, 'Initial learning rate')
  cmd:option('-weightDecay',         0, 'Weight decay')
  cmd:text(' ------------ Model options ----------------------')
  cmd:option('-netType',            '', 'Options: ')
  cmd:option('-inputRes',          256, 'Input image resolution')
  cmd:option('-outputRes',          64, 'Output heatmap resolution')
  cmd:option('-seqLength',          16, 'Sequence length')
  cmd:option('-hiddenSize',        256, 'Hidden state size')
  cmd:option('-numLayers',           1, 'Number of RNN layers') 
  cmd:option('-hgModel',        'none', 'Path to hourglass model for intialization')
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
