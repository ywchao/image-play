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
local DataLoader = torch.class('caffe-feat.DataLoader', M)

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
      -- This matters due to the thread-dependent randomness from data loading
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      return dataset:size()
   end

   local threads, outputs = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = outputs[1][1]
   self.batchSize = 1
end

function DataLoader:size()
   return math.ceil(self.__size)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.range(1, size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local input, imageSize
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  if not input then
                     imageSize = sample.input:size():totable()
                     table.remove(imageSize, 1)
                     input = torch.FloatTensor(sz, 10, unpack(imageSize))
                  end
                  input[i] = sample.input
               end
               collectgarbage()
               return {
                  index = indices:int(),
                  input = input:view(-1, unpack(imageSize)),
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