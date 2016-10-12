require 'lib/util/img'

M = {}

local function shuffleLR(x, matched_parts)
  local dim
  if x:nDimension() == 4 then
    dim = 2
  else
    assert(x:nDimension() == 3)
    dim = 1
  end

  for i = 1,#matched_parts do
    local idx1, idx2 = unpack(matched_parts[i])
    local tmp = x:narrow(dim, idx1, 1):clone()
    x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
    x:narrow(dim, idx2, 1):copy(tmp)
  end

  return x
end

function M.run(input, label, matched_parts)
  assert(#input == #label, 'input and label size mismatch')

  -- Color
  local m1 = torch.uniform(0.8, 1.2)
  local m2 = torch.uniform(0.8, 1.2)
  local m3 = torch.uniform(0.8, 1.2)
  for i = 1, #input do
    input[i][{1, {}, {}}]:mul(m1):clamp(0, 1)
    input[i][{2, {}, {}}]:mul(m2):clamp(0, 1)
    input[i][{3, {}, {}}]:mul(m3):clamp(0, 1)
  end

  -- Flip
  if torch.uniform() <= 0.5 then
    for i = 1, #input do
      input[i] = flip(input[i])
      label[i] = flip(shuffleLR(label[i], matched_parts))
    end
  end
end

return M