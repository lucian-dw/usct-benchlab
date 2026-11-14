from .config import AlgorithmConfig


def describe(variant=None):
    if variant not in (None, "cem", "continuum"):
        raise ValueError("variant must be cem or continuum")
    defaults = AlgorithmConfig().model_dump()
    if variant:
        defaults["parameters"]["variant"] = variant
    return dict(
        algorithm_id="lbfgsb_tv",
        description="EIT P1 FEM / discrete adjoint / L-BFGS-B + smoothed TV",
        modality="eit",
        family="nonlinear_variational",
        solver="L-BFGS-B",
        regularizer="smoothed_isotropic_fem_tv",
        mathematical_model="div(sigma grad u)=0; positive isotropic real conductivity; 2D P1 FEM",
        variants=["cem", "continuum"],
        modes=["difference", "absolute"],
        variant_models={
            "cem": "complete electrode model with physical contact impedance (ohm m^2)",
            "continuum": "insulated gaps, electrode piecewise sheet flux and average observation; boundary mean gauge",
        },
        unknown={"quantity": "conductivity", "units": "S/m", "domain": "mesh_nodes"},
        required_observation_domains=["boundary_voltage"],
        supported_geometries=["triangular_mesh_2d"],
        required_observations=[
            "currents_A",
            "measurement_matrix",
            "voltages_V",
            "reference_voltages_V",
            "measurement_mask",
        ],
        allowed_parameters=list(defaults["parameters"]),
        default_parameters=defaults["parameters"],
        default_run_controls=defaults["run_controls"],
        config_schema=AlgorithmConfig.model_json_schema(),
        budget_capabilities=[
            "max_iterations",
            "max_elapsed_s",
            "max_forward_calls",
            "max_adjoint_calls",
        ],
        stopping_conditions=[
            "projected_gradient_tolerance",
            "objective_tolerance",
            "max_iterations",
            "resource_exhausted",
            "numerical_error",
        ],
        budget_scope="calibration + baseline + reconstruction; one batched-RHS solve per forward/adjoint call; caches uncharged",
        parameter_source="supplied EIT submission defaults; no per-target tuning",
        runtime_requirements=["numpy", "scipy", "h5py", "pydantic", "PyYAML"],
        deterministic_status=True,
        limitations=[
            "CPU only",
            "nonconvex objective; local optimizer convergence is not reconstruction accuracy",
            "reference measurement required in both modes for calibration",
            "real photos are not quantitative conductivity truth",
            "no EIT neural solvers in this release",
            "time caps are cooperative between work units",
        ],
    )
