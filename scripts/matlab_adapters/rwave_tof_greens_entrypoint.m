% usct-benchlab wrapper for Ash1362/ray-based-quantitative-ultrasound-tomography.
%
% Expected variables from the Python adapter:
%   usctbench_input_mat
%   usctbench_output_mat
%   usctbench_output_dir
%   usctbench_parameters_json
%
% The public r-Wave toolbox exposes ToF reconstruction and Green's/ray-Born
% routines such as reconstructTimeofFlightImage.m and reconstructGreensImage.m.
% The standard usct-benchlab smoke cases often contain travel-time features,
% not full RF time series, so this wrapper produces the ToF initial image used
% by the r-Wave Green's workflow and writes it through the common adapter
% result contract.

case_data = usctbench_read_case(usctbench_input_mat);
if isempty(case_data.measurement.delta_tof_s)
    error('usctbench:missingDeltaTof', 'measurement.delta_tof_s is required');
end

shape = double(case_data.grid.shape);
ny = shape(1);
nx = shape(2);
c0 = usctbench_json_number(usctbench_parameters_json, 'reference_sound_speed_mps', 1500.0);
bounds = usctbench_json_vector(usctbench_parameters_json, 'sound_speed_bounds_mps', [1300.0, 1700.0]);
lambda = usctbench_json_number(usctbench_parameters_json, 'regularization_lambda', 1.0e-5);
num_cg = usctbench_json_number(usctbench_parameters_json, 'inner_iterations', 40);
smooth_sigma = usctbench_json_number(usctbench_parameters_json, 'smooth_sigma', 0.35);

target = double(case_data.measurement.delta_tof_s(:));
valid_mask = case_data.measurement.valid_mask;
if isempty(valid_mask)
    valid = true(size(target));
else
    valid = logical(valid_mask(:));
end

H = usctbench_build_straight_ray_matrix(case_data, nx, ny);
L = usctbench_laplacian_operator(nx, ny);
A = H(valid, :);
b = target(valid);
normal_matrix = A' * A + (lambda ^ 2) * (L' * L);
rhs = A' * b;
delta_slowness = pcg(normal_matrix, rhs, 1e-8, max(1, round(num_cg)));
if isempty(delta_slowness)
    delta_slowness = normal_matrix \ rhs;
end

slowness = reshape((1 / c0) + delta_slowness, [ny, nx]);
if smooth_sigma > 0
    slowness = usctbench_smooth2(slowness, smooth_sigma);
end
slowness = min(max(slowness, 1 / bounds(2)), 1 / bounds(1));
sound_speed = 1 ./ slowness;

metrics = sprintf(['{"external_entrypoint":"rwave_tof_greens_entrypoint",' ...
    '"external_reference":"Ash1362/ray-based-quantitative-ultrasound-tomography",' ...
    '"method_family":"external_rwave_tof_initialization",' ...
    '"ray_born_linearization":true,' ...
    '"regularization":"laplacian","regularization_lambda":%.17g,' ...
    '"smooth_sigma":%.17g}'], lambda, smooth_sigma);
usctbench_write_result(usctbench_output_mat, 'rwave_adapter', case_data.case_id, sound_speed, metrics);

function H = usctbench_build_straight_ray_matrix(case_data, nx, ny)
tx = double(case_data.geometry.tx_pos_m);
rx = double(case_data.geometry.rx_pos_m);
if size(tx, 2) ~= 2 && size(tx, 1) == 2
    tx = tx.';
end
if size(rx, 2) ~= 2 && size(rx, 1) == 2
    rx = rx.';
end
n_tx = size(tx, 1);
n_rx = size(rx, 1);
spacing = double(case_data.grid.spacing_m);
origin = double(case_data.grid.origin_m);
dx = spacing(2);
dy = spacing(1);
H = sparse(n_tx * n_rx, nx * ny);
row = 0;
for itx = 1:n_tx
    for irx = 1:n_rx
        row = row + 1;
        p0 = tx(itx, :);
        p1 = rx(irx, :);
        length_ray = norm(p1 - p0);
        samples = max(2, ceil(length_ray / (0.5 * min(dx, dy))));
        xs = linspace(p0(1), p1(1), samples);
        ys = linspace(p0(2), p1(2), samples);
        ix = round((xs - origin(2)) / dx) + 1;
        iy = round((ys - origin(1)) / dy) + 1;
        inside = ix >= 1 & ix <= nx & iy >= 1 & iy <= ny;
        if any(inside)
            cols = sub2ind([ny, nx], iy(inside), ix(inside));
            H(row, cols) = H(row, cols) + length_ray / max(1, numel(cols));
        end
    end
end
end

function L = usctbench_laplacian_operator(nx, ny)
idx = @(r, c) r + (c - 1) * ny;
[cols, rows] = meshgrid(1:nx, 1:ny);
rows = rows(:);
cols = cols(:);
center = idx(rows, cols);
left = idx(rows, max(1, cols - 1));
right = idx(rows, min(nx, cols + 1));
up = idx(max(1, rows - 1), cols);
down = idx(min(ny, rows + 1), cols);
L = sparse(center, center, -4, nx * ny, nx * ny);
L = L + sparse(center, left, 1, nx * ny, nx * ny);
L = L + sparse(center, right, 1, nx * ny, nx * ny);
L = L + sparse(center, up, 1, nx * ny, nx * ny);
L = L + sparse(center, down, 1, nx * ny, nx * ny);
end

function out = usctbench_smooth2(image, sigma)
radius = max(1, ceil(3 * sigma));
x = -radius:radius;
kernel = exp(-(x .^ 2) / (2 * sigma ^ 2));
kernel = kernel / sum(kernel);
out = conv2(conv2(image, kernel, 'same'), kernel.', 'same');
end

function value = usctbench_json_number(json_text, key, default_value)
value = default_value;
pattern = ['"' key '"\s*:\s*([-+0-9.eE]+)'];
tokens = regexp(json_text, pattern, 'tokens', 'once');
if ~isempty(tokens)
    value = str2double(tokens{1});
end
end

function value = usctbench_json_vector(json_text, key, default_value)
value = default_value;
pattern = ['"' key '"\s*:\s*\[([^\]]+)\]'];
tokens = regexp(json_text, pattern, 'tokens', 'once');
if ~isempty(tokens)
    parsed = sscanf(tokens{1}, '%f,');
    if numel(parsed) >= numel(default_value)
        value = parsed(1:numel(default_value)).';
    end
end
end
