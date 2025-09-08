function [total, primary] = read_mcgpu_image(filename, nx, nz)
%READ_MCGPU_IMAGE  Load MC-GPU projection images (total & primary) from a .raw file.
%
%   [TOTAL, PRIMARY] = READ_MCGPU_IMAGE(FILENAME, NX, NZ)
%
%   Inputs:
%     FILENAME - path to mcgpu_image.raw (or any MC-GPU .raw image file)
%     NX, NZ   - number of detector pixels in X and Z directions
%
%   Outputs:
%     TOTAL    - [nz x nx] image of primaries+scatter
%     PRIMARY  - [nz x nx] image of primaries only
%
%   Notes:
%     - Expects exactly 2 image planes, X-fastest layout.
%     - File must contain NX*NZ*2 float32 values (little-endian).
%     - Will error if file size does not match expected payload.

    if nargin < 3
        error('Usage: [total, primary] = read_mcgpu_image(filename, nx, nz)');
    end

    % Calculate expected payload
    nfloat = nx * nz * 2;
    nbytes = nfloat * 4;

    info = dir(filename);
    if isempty(info)
        error('File not found: %s', filename);
    end
    if info.bytes ~= nbytes
        warning('File size mismatch: expected %d bytes, found %d.', ...
            nbytes, info.bytes);
    end

    fid = fopen(filename, 'rb');
    if fid < 0
        error('Cannot open file: %s', filename);
    end

    data = fread(fid, nfloat, 'float32=>single', 0, 'ieee-le');
    fclose(fid);

    if numel(data) ~= nfloat
        error('Short read: expected %d floats, got %d.', nfloat, numel(data));
    end

    % Reshape: X-fastest, then transpose to [nz x nx]
    img = reshape(data, [nx, nz, 2]);
    total   = img(:,:,1)';  % Z-fastest
    primary = img(:,:,2)';
end
