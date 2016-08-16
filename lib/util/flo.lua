--
-- Utilities for manipulating flow files
--

-- local function convertint(str)
--   -- Change to b4, b3, b2, b1 to unpack an LSB float
--   b1, b2, b3, b4 = string.byte(str, 1, 4)
--   if not b4 then error("need four bytes to convert to int", 2) end
--   return b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
-- end

-- local function convertfloat(str)
--   -- Change to b4, b3, b2, b1 to unpack an LSB float
--   b1, b2, b3, b4 = string.byte(str, 1, 4)
--   exponent = (b4 % 128) * 2 + math.floor(b3 / 128)
--   if exponent == 0 then return 0 end
--   sign = (b4 > 127) and -1 or 1
--   mantissa = ((b3 % 128) * 256 + b2) * 256 + b1
--   mantissa = (math.ldexp(mantissa, -23) + 1) * sign
--   return math.ldexp(mantissa, exponent - 127)
-- end

-- local function convertfloat_batch(str)
--   if (str:len() % 4 ~= 0) then
--     error('convertfloat_batch: number of bytes should be divisible by 4')
--   end
--   local b = {string.byte(str, 1, #str)}
--   local t = torch.ByteTensor(b)
--   local b1 = t:index(1,torch.range(1,str:len(),4):type('torch.LongTensor')):type('torch.DoubleTensor')
--   local b2 = t:index(1,torch.range(2,str:len(),4):type('torch.LongTensor')):type('torch.DoubleTensor')
--   local b3 = t:index(1,torch.range(3,str:len(),4):type('torch.LongTensor')):type('torch.DoubleTensor')
--   local b4 = t:index(1,torch.range(4,str:len(),4):type('torch.LongTensor')):type('torch.DoubleTensor')
--   local exponent = (b4 % 128) * 2 + torch.floor(b3 / 128)
--   local sign = torch.ones(b1:size())
--   sign[b4:gt(127)] = -1
--   local mantissa = ((b3 % 128) * 256 + b2) * 256 + b1
--   mantissa = torch.cmul((mantissa*2^-23 + 1), sign)
--   local float = torch.cmul(mantissa,  torch.pow(2, exponent - 127))
--   float[exponent:eq(0)] = 0
--   return float
-- end

function readFlowFile(filename)
  local TAG_FLOAT = 202021.25
  
  -- sanity check
  if filename == nil then
      error('readFlowFile: empty filename')
  end
  if string.sub(filename,-4,-1) ~= '.flo' then
      error(string.format('readFlowFile: filename %s should have extension ".flo"', filename))
  end

  -- local fid = io.open(filename, "rb")
  -- if fid == nil then
  --     error(string.format('readFlowFile: could not open %s', filename))
  -- end
  local ff = torch.DiskFile(filename)
  ff:binary()

  -- local tag = convertfloat(fid:read(4))
  -- local width = convertint(fid:read(4))
  -- local height = convertint(fid:read(4))
  local tag = ff:readFloat()
  local width = ff:readInt()
  local height = ff:readInt()
  
  -- sanity check
  if (tag ~= TAG_FLOAT) then
    error(string.format('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)', filename));
  end
  if (width < 1 or width > 99999) then
    error(string.format('readFlowFile(%s): illegal width %d', filename, width))
  end
  if (height < 1 or height > 99999) then
    error(string.format('readFlowFile(%s): illegal height %d', filename, height))
  end

  local nBands = 2;

  -- -- arrange into matrix form
  -- local tmp = fid:read('*all')

  -- if (tmp:len() % 4 ~= 0) then
  --   error(string.format('readFlowFile(%s): number of bytes should be divisible by 4', filename))
  -- end
  -- local n1 = tmp:len()/4
  -- local n2 = width * height * nBands
  -- if (n1 ~= n2) then
  --   error(string.format('readFlowFile(%s): incorrect number of pixels (%d vs. %d)', filename, n1, n2))
  -- end

  -- local img = torch.Tensor(height*width*nBands)
  -- for i = 1, n1 do
  --   local str = string.sub(tmp,(i-1)*4+1,i*4)
  --   img[i] = convertfloat(str)
  -- end
  -- img = img:view(height, width, nBands)

  -- batch mode: slightly faster, but still no compare with C version; might require further optimization later
  -- chunk_size = 2500
  -- img = torch.Tensor(height*width*nBands)
  -- for i = 1, math.ceil(tmp:len() / (chunk_size * 4)) do
  --   str = string.sub(tmp,(i-1)*chunk_size*4+1,math.min(i*chunk_size*4,tmp:len()))
  --   img:sub((i-1)*chunk_size+1,math.min(i*chunk_size,tmp:len()/4)):copy(convertfloat_batch(str))
  -- end
  -- img = img:view(height, width, nBands)

  -- fid:close()

  local img = torch.FloatTensor(height * width * nBands)
  ff:readFloat(img:storage())
  img = img:view(height, width, nBands)

  return img
end
