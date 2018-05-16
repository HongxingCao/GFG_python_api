function out = run_GFG(face_id,nfdata,save_path,au_labels,temp_params,eye_params,head_params,version)
% run_GFG Animates a face from the database with given inputs.

% Transform ndarray to cell (as expected by toolbox)
temp_params2 = {
    temp_params(1, :);
    temp_params(2, :);
    temp_params(3, :);
};

if ischar(nfdata)
    nfname = strrep(nfdata, '.mat', '');
    [~,nfname,~] = fileparts(nfname);
    nfdata = load(nfdata);
    
    if ~strcmp(version, 'v1')
        error('Cannot generate faces (textures) other than in v1!');
    end
end

% check which shapes to load
if strcmp(version, 'v2dense')
    adata = default_FACS_blend_shapes_v2dense;
elseif strcmp(version, 'v2')
    adata = default_FACS_blend_shapes_v2;
else
    adata = default_FACS_blend_shapes;
end

% Create AUset to animate
AUset = [];
AUset.adata = adata;
clear adata

if strcmp(version, 'v2dense')
    if face_id == 0
        nf = load('default_face_v2dense');
    else
        nf = load_face_v2dense(face_id);
    end
elseif strcmp(version, 'v2')
    if face_id == 0
        nf = load('default_face_v2');
    else
        nf = load_face_v2(face_id);
    end
else
    if face_id == 0
        nf = load('default_face');
    else
        nf = load_face(face_id);
    end
end

if isstruct(nfdata)
    this_save_path = fullfile(save_path, nfname);
else
    this_save_path = fullfile(save_path, ['id-' num2str(face_id)]);
end

if ~exist(this_save_path, 'dir')
    mkdir(this_save_path);
end

AUset.nf = nf;

if isstruct(nfdata)
    AUset.nf.v = nfdata.vertices;
    AUset.nf.texture = nfdata.textures;
end

AUset.savepath = this_save_path;
AUset.labels = au_labels;
AUset.params = temp_params2;
AUset.eyeparams = eye_params;

if head_params ~= 0
    AUset.headparams = head_params;
end

% Start animation!
AUset = animate_AUset_GL(AUset);

% Status code
out = 0;

end
