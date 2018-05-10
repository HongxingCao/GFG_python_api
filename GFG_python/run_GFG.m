function out = run_GFG(face_id,save_path,au_labels,temp_params,eye_params,head_params,v2,dense)
% run_GFG Animates a face from the database with given inputs.

% Transform ndarray to cell (as expected by toolbox)
temp_params2 = {
    temp_params(1, :);
    temp_params(2, :);
    temp_params(3, :);
};

% check which shapes to load
if v2 == 1
    if dense == 1
        fprintf('Loading shapes v2 dense\n')
        adata = default_FACS_blend_shapes_v2dense;
    else
        fprintf('Loading shapes v2\n')
        adata = default_FACS_blend_shapes_v2;
    end
else
    fprintf('Loading shapes v1\n')
    adata = default_FACS_blend_shapes;
end

% Create AUset to animate
AUset = [];
AUset.adata = adata;
clear adata

% Loop over face-ids
for i = 1:length(face_id)
    this_id = face_id(i);
    fprintf('Processing id: %i\n', this_id);

    if v2 == 1
        if dense == 1
            nf = load_face_v2dense(this_id);
        else
            nf = load_face_v2(this_id);
        end
    else
        nf = load_face(this_id);
    end

    this_save_path = fullfile(save_path, num2str(this_id));
    if ~exist(this_save_path, 'dir')
        mkdir(this_save_path);
    end

    AUset.nf = nf;
    AUset.savepath = this_save_path;
    AUset.labels = au_labels;
    AUset.params = temp_params2;
    AUset.eyeparams = eye_params;

    if head_params ~= 0
        AUset.headparams = head_params;
    end

    % Start animation!
    AUset = animate_AUset_GL(AUset);
end

% Status code
out = 0;

end
