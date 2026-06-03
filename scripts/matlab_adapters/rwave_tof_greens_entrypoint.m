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
% This wrapper is contract-aware:
%   1. If the Python adapter exported a /rwave complex-wavefield contract, use
%      the Rytov phase-slope target plus complex quality weights.
%   2. Otherwise fall back to the historical ToF smoke path.
% It is still not the full reconstructGreensImage path unless the external
% repository is wired to consume the full time-domain Green's workflow.

case_data = usctbench_read_case(usctbench_input_mat);

shape = double(case_data.grid.shape);
ny = shape(1);
nx = shape(2);
c0 = usctbench_json_number(usctbench_parameters_json, 'reference_sound_speed_mps', 1500.0);
bounds = usctbench_json_vector(usctbench_parameters_json, 'sound_speed_bounds_mps', [1300.0, 1700.0]);
lambda = usctbench_json_number(usctbench_parameters_json, 'regularization_lambda', 1.0e-5);
num_cg = usctbench_json_number(usctbench_parameters_json, 'inner_iterations', 40);
smooth_sigma = usctbench_json_number(usctbench_parameters_json, 'smooth_sigma', 0.35);
outer_iterations = usctbench_json_number(usctbench_parameters_json, 'outer_iterations', 3);
step_length = usctbench_json_number(usctbench_parameters_json, 'step_length', 1.0);
roi_update_only = usctbench_json_bool(usctbench_parameters_json, 'roi_update_only', false);

use_complex_contract = isfield(case_data, 'rwave') && ~isempty(case_data.rwave.phase_slope_delay_s);
if use_complex_contract
    target = double(case_data.rwave.phase_slope_delay_s(:));
    valid_mask = case_data.rwave.complex_valid_mask;
    if isempty(valid_mask)
        valid = isfinite(target);
    else
        valid = logical(valid_mask(:)) & isfinite(target);
    end
    ray_weights = case_data.rwave.complex_quality;
    if isempty(ray_weights)
        ray_weights = ones(size(target));
    else
        ray_weights = double(ray_weights(:));
    end
    if numel(ray_weights) ~= numel(target)
        ray_weights = ones(size(target));
    end
    method_family = 'external_rwave_complex_contract_phase_slope';
    complex_text = 'true';
    surrogate_text = 'false';
    target_source = 'rwave_complex_contract';
else
    if isempty(case_data.measurement.delta_tof_s)
        error('usctbench:missingInput', 'measurement.delta_tof_s or /rwave/phase_slope_delay_s is required');
    end
    target = double(case_data.measurement.delta_tof_s(:));
    valid_mask = case_data.measurement.valid_mask;
    if isempty(valid_mask)
        valid = true(size(target));
    else
        valid = logical(valid_mask(:));
    end
    ray_weights = case_data.measurement.ray_weights;
    if isempty(ray_weights)
        ray_weights = case_data.measurement.feature_quality;
    end
    if isempty(ray_weights)
        ray_weights = ones(size(target));
    else
        ray_weights = double(ray_weights(:));
    end
    if numel(ray_weights) ~= numel(target)
        ray_weights = ones(size(target));
    end
    method_family = 'external_rwave_tof_initialization';
    complex_text = 'false';
    surrogate_text = 'true';
    target_source = 'measurement_delta_tof';
end

H = usctbench_build_straight_ray_matrix(case_data, nx, ny);
L = usctbench_laplacian_operator(nx, ny);
roi = usctbench_roi_mask(case_data, nx, ny);
A = H(valid, :);
b = target(valid);
w = max(0, min(1, ray_weights(valid)));
sqrt_w = sqrt(w);
A_weighted = spdiags(sqrt_w, 0, length(sqrt_w), length(sqrt_w)) * A;
b_weighted = sqrt_w .* b;
normal_matrix = A_weighted' * A_weighted + (lambda ^ 2) * (L' * L);
delta_slowness = zeros(nx * ny, 1);
for iter = 1:max(1, round(outer_iterations))
    residual = b_weighted - A_weighted * delta_slowness;
    rhs = A_weighted' * residual;
    update = pcg(normal_matrix, rhs, 1e-8, max(1, round(num_cg)));
    if isempty(update)
        update = normal_matrix \ rhs;
    end
    delta_slowness = delta_slowness + step_length * update;
    if smooth_sigma > 0
        delta_slowness = reshape(usctbench_smooth2(reshape(delta_slowness, [ny, nx]), smooth_sigma), [], 1);
    end
    if roi_update_only
        delta_image = reshape(delta_slowness, [ny, nx]);
        delta_image(~roi) = 0;
        delta_slowness = delta_image(:);
    end
    slowness_iter = min(max((1 / c0) + reshape(delta_slowness, [ny, nx]), 1 / bounds(2)), 1 / bounds(1));
    if roi_update_only
        slowness_iter(~roi) = 1 / c0;
    end
    delta_slowness = reshape(slowness_iter - (1 / c0), [], 1);
end

slowness = reshape((1 / c0) + delta_slowness, [ny, nx]);
slowness = min(max(slowness, 1 / bounds(2)), 1 / bounds(1));
if roi_update_only
    slowness(~roi) = 1 / c0;
end
sound_speed = 1 ./ slowness;

metrics = sprintf(['{"external_entrypoint":"rwave_tof_greens_entrypoint",' ...
    '"external_reference":"Ash1362/ray-based-quantitative-ultrasound-tomography",' ...
    '"method_family":"%s",' ...
    '"target_source":"%s",' ...
    '"ray_born_linearization":true,' ...
    '"uses_complex_wavefield":%s,' ...
    '"complex_contract_used":%s,' ...
    '"external_greens_full_wavefield":false,' ...
    '"surrogate_travel_time_backend":%s,' ...
    '"complex_valid_fraction":%.17g,' ...
    '"regularization":"laplacian","regularization_lambda":%.17g,' ...
    '"outer_iterations":%.17g,"inner_iterations":%.17g,' ...
    '"smooth_sigma":%.17g,"roi_update_only":%s}'], method_family, target_source, complex_text, complex_text, surrogate_text, ...
    mean(valid), lambda, outer_iterations, num_cg, smooth_sigma, usctbench_bool_text(roi_update_only));
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

function roi = usctbench_roi_mask(case_data, nx, ny)
if isfield(case_data, 'grid') && isfield(case_data.grid, 'roi_mask') && ~isempty(case_data.grid.roi_mask)
    roi = logical(case_data.grid.roi_mask);
    if ~isequal(size(roi), [ny, nx])
        roi = reshape(roi, [ny, nx]);
    end
else
    roi = true(ny, nx);
end
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

function value = usctbench_json_bool(json_text, key, default_value)
value = default_value;
pattern = ['"' key '"\s*:\s*(true|false|1|0)'];
tokens = regexp(json_text, pattern, 'tokens', 'once');
if ~isempty(tokens)
    token = lower(tokens{1});
    value = strcmp(token, 'true') || strcmp(token, '1');
end
end

function text = usctbench_bool_text(value)
if value
    text = 'true';
else
    text = 'false';
end
end
