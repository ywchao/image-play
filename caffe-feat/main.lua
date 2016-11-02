require 'cutorch'

local DataLoader = require 'lib/dataloader'
local models = require 'lib/models/init'
local Trainer = require 'lib/train'
local opts = require 'lib/opts'

local opt = opts.parse(arg)

cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Create model
local model = models.setup(opt)

-- Data loading
local loaders = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, opt)

-- Run prediction
trainer:predict(loaders, 'train')
trainer:predict(loaders, 'val')
trainer:predict(loaders, 'test')