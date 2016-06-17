
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
