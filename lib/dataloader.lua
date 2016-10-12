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
      loaders[split] = M.DataLoader(opt, split)
   end

   -- return table.unpack(loaders)
   return loaders
end

-- function DataLoader:__init(dataset, opt, split)
function DataLoader:__init(opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      -- require('lib/datasets/' .. opt.dataset)
      -- We should have initialize dataset in creat(). This is currently not
      -- possible since the used hdf5 library will throw errors if we do that.
      local Dataset = require('lib/datasets/' .. opt.dataset)
      dataset = Dataset(opt, split)
      augment = require('lib/augment')
   end
   local function main(idx)
      -- This matters due to the thread-dependent randomness from data
      -- augmentation
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.augment = augment
      return dataset:size(), dataset:getSampledIdx()
   end

   local threads, outputs = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = outputs[1][1]
   self.sidx = outputs[1][2]
   if split == 'train' then
     self.batchSize = opt.batchSize
   else
     self.batchSize = 1
   end
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
   if kwargs ~= nil and kwargs.pred == true then
      batchSize = 1
      perm = torch.range(1, size)
   else
      perm = torch.randperm(size)
   end
   if kwargs ~= nil and kwargs.samp == true then
      assert(kwargs.pred == true, 'kwargs error: require pred == true if samp == true')
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
               -- local input = {}
               -- local target = {}
               -- for i, idx in ipairs(indices:totable()) do
               --    sample = _G.dataset:get(idx)
               --    input[i] = sample.input
               --    target[i] = sample.target
               -- end
               -- collectgarbage()
               -- return {
               --    input = input,
               --    target = target,
               -- }
               local sz = indices:size(1)
               local index = {}
               local input, imageSize
               local target_ps, targetSizes
               for i, idx in ipairs(indices:totable()) do
                  index[i] = idx
                  local sample = _G.dataset:get(idx)
                  -- Augment data
                  if kwargs ~= nil and kwargs.augment == true then
                     _G.augment.run(sample.input, sample.target, _G.dataset:getMatchedParts())
                  end
                  if not input then
                     -- imageSize = sample.input:size():totable()
                     -- input = torch.FloatTensor(sz, unpack(imageSize))
                     imageSize = sample.input[1]:size():totable()
                     input = {}
                     for j = 1, #sample.input do
                        input[j] = torch.FloatTensor(sz, unpack(imageSize))
                     end
                  end
                  if not target_ps then
                     targetSize = sample.target[1]:size():totable()
                     target_ps = {}
                     for j = 1, #sample.target do
                        target_ps[j] = torch.FloatTensor(sz, unpack(targetSize))
                     end
                  end
                  -- input[i]:copy(sample.input)
                  for j = 1, #input do
                     input[j][i] = sample.input[j]
                  end
                  for j = 1, #target_ps do
                     target_ps[j][i] = sample.target[j]
                  end
               end
               collectgarbage()
               return {
                  index = index,
                  input = input,
                  target_ps = target_ps,
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
