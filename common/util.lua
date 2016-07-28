
function dir(path, ext)
  list = {}
  for file in paths.files(path) do
    if file:find(ext .. '$') then
      table.insert(list, file)
    end
  end
  table.sort(list)
  return list
end

function makedir(dir)
  if not paths.dirp(dir) then
    paths.mkdir(dir)
  end
end

function keys(table)
  local keys={}
  local i = 0
  for k in pairs(table) do
    i = i+1
    keys[i] = k
  end
  return keys
end

function unique(input)
-- input should be a one dimensinoal tensor
  local b = {}
  for i = 1, input:numel() do
    b[input[i]] = true
  end
  local out = {}
  for i in pairs(b) do
    table.insert(out, i)
  end
  table.sort(out)
  return torch.Tensor(out)
end

function find(X, n)
-- X should be a one dimensional tensor
  local indices = torch.linspace(1,X:size(1),X:size(1)):long()
  return indices[X:eq(n)]
end
