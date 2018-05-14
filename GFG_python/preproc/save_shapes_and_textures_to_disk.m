% Write vertices to disk
cinfo = get_capture_info;
N_v1 = sum(cinfo.proc);
N_v2 = sum(cinfo.proc_v2);

vertices_v1 = zeros(4735, 3, N_v1);
textures_v1 = zeros(800, 600, 4, N_v1);
vertices_v2 = zeros(13780, 3, N_v2);
vertices_v2dense = zeros(32493, 3, N_v2);

i_v1 = 1;
i_v2 = 1;
for i = 1:length(cinfo.scode)
    scode = cinfo.scode(i);

    if cinfo.proc(i) == 1
        fprintf('V1: processing %i/%i\n', [i_v1, N_v1]);
        this_face = load_face(scode);
        vertices_v1(:, :, i_v1) = this_face.v;
        textures_v1(:, :, :, i_v1) = this_face.texture;
        i_v1 = i_v1 + 1;
        h5create('/Users/lukas/desktop/testh5.h5', ['/v1/' num2str(scode) '/vertices'], size(this_face.v));
        h5write('/Users/lukas/desktop/testh5.h5', ['/v1/' num2str(scode) '/vertices'], this_face.v);
        h5create('/Users/lukas/desktop/testh5.h5', ['/v1/' num2str(scode) '/textures'], size(this_face.texture));
        h5write('/Users/lukas/desktop/testh5.h5', ['/v1/' num2str(scode) '/textures'], this_face.texture);
    end

    if cinfo.proc_v2(i) == 1
        fprintf('V2: processing %i/%i\n', [i_v2, N_v2]);
        this_face = load_face_v2(scode);
        vertices_v2(:, :, i_v2) = this_face.v;
        this_face = load_face_v2dense(scode);
        vertices_v2dense(:, :, i_v2) = this_face.v;
        i_v2 = i_v2 + 1;
        h5create('/Users/lukas/desktop/testh5.h5', ['/v2/' num2str(scode) '/vertices'], size(this_face.v));
        h5write('/Users/lukas/desktop/testh5.h5', ['/v2/' num2str(scode) '/vertices'], this_face.v);
    end
end
