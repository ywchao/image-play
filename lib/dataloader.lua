--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('image-play.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val', 'test'} do
      -- local dataset = Dataset(opt, split)
      -- loaders[split] = M.DataLoader(dataset, opt, split)
      -- loaders[split] = M.DataLoader(opt, split)
      local Dataset = require('lib/datasets/' .. opt.dataset)
      local dataset = Dataset(opt, split)
      loaders[split] = M.DataLoader(dataset, opt, split)
   end

   -- return table.unpack(loaders)
   return loaders
end

function DataLoader:__init(dataset, opt, split)
-- function DataLoader:__init(opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      -- require('lib/datasets/' .. opt.dataset)
      -- We should have initialize dataset in creat(). This is currently not
      -- possible since the used hdf5 library will throw errors if we do that.
      local Dataset = require('lib/datasets/' .. opt.dataset)
      -- dataset = Dataset(opt, split)
   end
   local function main(idx)
      -- This matters due to the thread-dependent randomness from data
      -- augmentation
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      return dataset:size(), dataset:getSampledIdx()
   end

   local threads, outputs = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = outputs[1][1]
   self.sidx = outputs[1][2]
   self.batchSize = opt.batchSize
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:sizeDataset()
   return self.__size
end

function DataLoader:sizeSampled()
   return self.sidx:numel()
end

function DataLoader:run(kwargs)
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm
   assert(kwargs ~= nil and kwargs.train ~= nil)
   if kwargs.train == true then
      perm = torch.randperm(size)
   else
      batchSize = 1
      perm = torch.range(1, size)
   end
   if kwargs.samp == true then
      assert(kwargs.train == false, 'kwargs error: require train == false if samp == true')
      batchSize = 1
      perm = self.sidx
      size = self.sidx:numel()
   end

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local input, imageSize
               local repos, reposSize
               local trans, transSize
               local focal
               local hmap, hmapSize
               local proj, projSize
               local gtpts, gtptsSize
               local center
               local scale
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx, kwargs.train)
                  if not input then
                     imageSize = sample.input[1]:size():totable()
                     reposSize = sample.repos[1]:size():totable()
                     transSize = sample.trans[1]:size():totable()
                     hmapSize = sample.hmap[1]:size():totable()
                     projSize = sample.proj[1]:size():totable()
                     gtptsSize = sample.proj[1]:size():totable()
                     input, repos, trans, focal, hmap, proj, gtpts, center, scale =
                         {}, {}, {}, {}, {}, {}, {}, {}, {}
                     for j = 1, #sample.input do
                        input[j] = torch.FloatTensor(sz, unpack(imageSize))
                        repos[j] = torch.FloatTensor(sz, unpack(reposSize))
                        trans[j] = torch.FloatTensor(sz, unpack(transSize))
                        focal[j] = torch.FloatTensor(sz, 1)
                        hmap[j] = torch.FloatTensor(sz, unpack(hmapSize))
                        proj[j] = torch.FloatTensor(sz, unpack(projSize))
                        gtpts[j] = torch.FloatTensor(sz, unpack(gtptsSize))
                     end
                     center = torch.FloatTensor(sz, 2)
                     scale = torch.FloatTensor(sz, 1)
                  end
                  for j = 1, #input do
                     input[j][i] = sample.input[j]
                     repos[j][i] = sample.repos[j]
                     trans[j][i] = sample.trans[j]
                     focal[j][i] = sample.focal[j]
                     hmap[j][i] = sample.hmap[j]
                     proj[j][i] = sample.proj[j]
                     gtpts[j][i] = sample.gtpts[j]
                  end
                  center[i] = sample.center
                  scale[i] = sample.scale
               end
               collectgarbage()
               return {
                  index = indices:int(),
                  input = input,
                  repos = repos,
                  trans = trans,
                  focal = focal,
                  hmap = hmap,
                  proj = proj,
                  gtpts = gtpts,
                  center = center,
                  scale = scale,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader