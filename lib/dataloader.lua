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
               local label, labelSizes
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx, kwargs.train)
                  if not input then
                     imageSize = sample.input[1]:size():totable()
                     input = {}
                     for j = 1, #sample.input do
                        input[j] = torch.FloatTensor(sz, unpack(imageSize))
                     end
                  end
                  if not label then
                     labelSize = sample.label[1]:size():totable()
                     label = {}
                     for j = 1, #sample.label do
                        label[j] = torch.FloatTensor(sz, unpack(labelSize))
                     end
                  end
                  for j = 1, #input do
                     input[j][i] = sample.input[j]
                  end
                  for j = 1, #label do
                     label[j][i] = sample.label[j]
                  end
               end
               collectgarbage()
               return {
                  index = indices:int(),
                  input = input,
                  label = label,
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