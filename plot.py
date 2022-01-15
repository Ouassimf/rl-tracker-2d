import os
from stable_baselines3.common import results_plotter
from matplotlib import pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

timesteps = 2000000

log_dir = os.path.join(os.getcwd(),"run_mv_fast_target_w_dist_double_obs")
os.makedirs(log_dir, exist_ok=True)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Tracking 2D")
plt.show()
