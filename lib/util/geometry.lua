local M = {}

function M.angle2dcm(r1, r2, r3, S)
  S = S or 'zyx'

  local dcm = torch.zeros(3,3)
  local cang = torch.cos(torch.Tensor{r1,r2,r3})
  local sang = torch.sin(torch.Tensor{r1,r2,r3})

  -- now only supports 'zyx'
  assert(S:lower() == 'zyx')

  if S:lower() == 'zyx' then
    dcm[1][1] = cang[2] * cang[1]
    dcm[1][2] = cang[2] * sang[1]
    dcm[1][3] = -sang[2]
    dcm[2][1] = sang[3] * sang[2] * cang[1] - cang[3] * sang[1]
    dcm[2][2] = sang[3] * sang[2] * sang[1] + cang[3] * cang[1]
    dcm[2][3] = sang[3] * cang[2]
    dcm[3][1] = cang[3] * sang[2] * cang[1] + sang[3] * sang[1]
    dcm[3][2] = cang[3] * sang[2] * sang[1] - sang[3] * cang[1]
    dcm[3][3] = cang[3] * cang[2]
  end

  return dcm
end

function M.camProject(P, R, T, f, c)
-- input
--	P:	N x 3
--	R:	3 x 3
--	T:	1 x 3
--	f:	1 x 2
--	c:	1 x 2
-- output
--	p:	N x 2
--	D:	N x 1
--  X:  N x 3
  local N = P:size(1)
  local X = R * torch.csub(P:t(),T:expand(N,3):t())
  local p = torch.cdiv(X[{{1,2},{}}], X[{{3},{}}]:expand(2,N))
  local p = torch.cmul(f:expand(N,2), p:t()) + c:expand(N,2)
  local D = X[{{3}}]:t()
  local X = X:t()
  return p, D, X
end

function M.shuffleLR(x, dataset)
  assert(x:dim() == 3 or x:dim() == 2, 'dim must be 3 or 2')
  local dim = x:dim() - 1
  local matchedParts
  if dataset == 'penn-crop' then
    matchedParts = {{ 2, 3}, { 4, 5}, { 6, 7}, { 8, 9}, {10,11}, {12,13}}
  end
  local y = x:clone()
  for i = 1, #matchedParts do
    local idx1, idx2 = unpack(matchedParts[i])
    y:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
    y:narrow(dim, idx2, 1):copy(x:narrow(dim, idx1, 1))
  end
  return y
end

function M.flip(x)
  local dim = x:dim()
  assert(dim == 3 or dim == 2, 'dim must be 3 or 2')
  assert(x:size(dim) == 3 or x:size(dim) == 2, 'last dim must be 2D or 3D')
  local y = x:clone()
  y:select(dim,1):mul(-1)
  return y
end

return M