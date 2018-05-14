% Saves all fields separately to disk to be loaded in into Python
cinfo = get_capture_info;

all_fieldnames = fieldnames(cinfo);
for i = 1:size(all_fieldnames)
    this_name = all_fieldnames(i);
    this_field = cinfo.(this_name{1});
    save(['/Users/lukas/software/GFG_python/GFG_python/data/', this_name{1}, '.mat'], 'this_field');
end