from sarpy_plus.targets import generate_scatterers_from_model, plot_scatterers_3d
from sarpy_plus import (RadarParams,
                        TargetParams,
                        SAR_Sim,
                        plot_time_2d,
                        plot_space_2d,
                        rda,
                        wka)

# Generate from model
target = generate_scatterers_from_model('Cybertruck.obj', num_centers=500, rcs_scale=20.0, edge_bias=1.0, edge_rcs_boost=1.5, aspect_angle_deg=90.0)  # dB edges



plot_scatterers_3d(target)

radar = RadarParams(
    platform_altitude_m=200.0,
    platform_speed_mps=100.0,
    center_frequency_hz=10e9,
    range_resolution_m=0.25,
    cross_range_resolution_m=0.25,
    pulse_width_sec=0.25e-6,
    prf_hz=5000.0,
    range_oversample=4.0,
    ground_range_swath_m=100.0,
    range_grp_m=500.0,
    azimuth_aperture_factor=1.0,
    SNR_SAR=-10.0,
    antenna_pattern="spotlight",
    noise=True
)


ph = SAR_Sim(radar, target)

plot_time_2d(ph, radar)

image = wka(ph, radar)

plot_space_2d(image, radar)