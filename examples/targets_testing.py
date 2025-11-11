from sarpy_plus.targets import generate_scatterers_from_model, plot_scatterers_3d



# Generate from model
target = generate_scatterers_from_model('Cybertruck.obj', num_centers=500, rcs_scale=30.0, edge_bias=0.0, edge_rcs_boost=1.5)  # dB edges



plot_scatterers_3d(target)

# Sim as usual
# radar = RadarParams(antenna_pattern="parabolic", beamwidth_deg=1.2)
# raw = SAR_Sim(radar, target, noise_db=-10)
