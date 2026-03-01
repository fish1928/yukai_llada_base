import torch

# write a same-idx-for-rows version first
# [current] = [refresh|denoising
def concat_and_replace(matrix_origin, matrix_current, idx_refresh, idx_denoising, shape_target): # (B, L, H)

    if matrix_origin.shape[1] < shape_target[1]:   # need patch
        length_patch = shape_target[1] - matrix_origin.shape[1]

        assert matrix_current.shape[1] >= length_patch,\
            f'current shape should be >= patch shape, {matrix_current.shape[1]} >= {length_patch}'
        matrix_patch = matrix_current[:, -length_patch:, :]   # TODO: check this
        print(matrix_origin.shape, matrix_patch.shape)
        matrix_origin = torch.cat([matrix_origin, matrix_patch], dim=1)
    # end

    assert matrix_origin.shape[1] == shape_target[1],\
        f'origin shape should equal to target shape after patch, {matrix_origin.shape[1]} == {shape_target[1]}'

    idx_current = torch.cat([idx_refresh, idx_denoising], dim=1)
    matrix_origin[:, idx_current, :] = matrix_current
    return matrix_origin
# end