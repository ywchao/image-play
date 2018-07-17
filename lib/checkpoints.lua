--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
   if not opt.resume then
      return nil, nil, opt
   end

   local latestPath = paths.concat(opt.save, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil, nil, opt
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.save, latest.optimFile))

   -- Get manually set options
   local setOpts = {}
   for i = 1, #arg do
      if arg[i]:sub(1,1) == '-' then
         table.insert(setOpts,arg[i]:sub(2,-1))
      end
   end

   -- Keep previous option, except those that are manually set
   local opt_ = opt
   opt = torch.load(paths.concat(opt.save, latest.optFile))
   opt.resume = true
   for i = 1, #setOpts do
      opt[setOpts[i]] = opt_[setOpts[i]]
   end

   return latest, optimState, opt
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt)
   -- Remove temporary buffers to reduce checkpoint size
   model:clearState()

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'
   local optFile = 'options.t7'

   torch.save(paths.concat(opt.save, modelFile), model)
   torch.save(paths.concat(opt.save, optimFile), optimState)
   torch.save(paths.concat(opt.save, optFile), opt)
   torch.save(paths.concat(opt.save, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
      optFile = optFile,
   })

   if isBestModel then
      torch.save(paths.concat(opt.save, 'model_best.t7'), model)
   end
end

return checkpoint