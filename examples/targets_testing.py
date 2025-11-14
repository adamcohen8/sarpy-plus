from sarpy_plus.targets import plot_scatterers_3d, generate_scatterers_from_model
from sarpy_plus import (RadarParams,
                        TargetParams,
                        SAR_Sim,
                        plot_time_2d,
                        plot_space_2d,
                        rda,
                        wka)

# Generate from model

target, bvh, meta = generate_scatterers_from_model(
    "Cybertruck.obj",
    num_centers=256,
    orient="auto",               # ← auto-detect up = Z, length = Y
    subdivide_levels=3,
    edge_fraction=0.50,
    corner_fraction=0.10,
    min_per_face=3,
    surface_edge_hug_frac=0.55,
    hard_edge_threshold_deg=6.0,
    silhouette_boost_frac=0.45,
    silhouette_bound_eps=0.06,
    jitter_tangent=0.0010,
    octant_floor_surface_frac=0.06,
    octant_floor_edge_frac=0.06,
    random_seed=17
)


plot_scatterers_3d(target)

radar = RadarParams(
    platform_altitude_m=500.0,
    platform_speed_mps=100.0,
    center_frequency_hz=15e9,
    range_resolution_m=0.1,
    cross_range_resolution_m=0.1,
    pulse_width_sec=0.25e-6,
    prf_hz=3000.0,
    range_oversample=2.0,
    ground_range_swath_m=100.0,
    range_grp_m=500.0,
    azimuth_aperture_factor=1.0,
    SNR_SAR=50.0,
    antenna_pattern="spotlight",
    noise=False
)


ph = SAR_Sim(radar, target, bvh=bvh, meta=meta)

plot_time_2d(ph, radar)

image = rda(ph, radar)

plot_space_2d(image, radar, window=8)
plot_space_2d(image, radar, window=8, db=True)