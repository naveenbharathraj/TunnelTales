import numpy as np
import scipy.io as spio
from scipy.interpolate import UnivariateSpline
from lmfit import Model
import time
import logging
from src.solver_configuration.solver_configuration import SolverConfiguration
from src import solver

# Setup logging configuration
logging.basicConfig(filename='log/output.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.INFO)

# Define the objective function
def objective(x, c_head, c_tail, train_tunnel_friction, tunnel_friction, 
              c_portal, cc, cc1, u_right, rho_right):
    # Update rd parameters
    rd.c_head, rd.c_tail = c_head, c_tail
    rd.train_tunnel_friction, rd.tunnel_friction = train_tunnel_friction, tunnel_friction
    rd.c_portal, rd.cc, rd.cc1 = c_portal, cc, cc1
    rd.u_right, rd.rho_right = u_right, rho_right

    print_tuning_variables()
    # Run the solver and calculate the error
    p_history, _ = solver.solver(rd, logging)
    return (p_history[0, :] - rd.pressure_right) / 1000

def print_tuning_variables():
    for attr in ['c_head', 'c_tail', 'train_tunnel_friction', 'tunnel_friction', 
                 'c_portal', 'cc', 'cc1', 'u_right', 'rho_right']:
        print(f"{attr} : {getattr(rd, attr)}")
        logging.info(f"{attr} : {getattr(rd, attr)}")

# Load configuration
file_path = "input/train_a.cfg"
rd = SolverConfiguration(file_path, False)
print('Configuration Loaded for Velocity:', rd.train_velocity)
logging.info('Loaded Configuration for Velocity: %s', rd.train_velocity)

# Load experimental data
data = spio.loadmat('output/experiment/P1.mat')
train_no = 0
x_data, y_data = data['xdata'][train_no][:], data['ydata'][train_no][:]
del data
print('Experiment Data Loaded')

# Interpolation
interp_func = UnivariateSpline(x_data, y_data)
exp_y_inp = interp_func(np.arange(0, rd.total_time, rd.dt)) / 1000
print('Interpolation Completed')

# Define the model and parameters
mymodel = Model(objective)
param_settings = {
    "c_head": (rd.c_head, 0.1, 100),
    "c_tail": (rd.c_tail, 0.1, 100),
    "train_tunnel_friction": (rd.train_tunnel_friction, 1e-5, 1),
    "tunnel_friction": (rd.tunnel_friction, 1e-5, 1),
    "c_portal": (rd.c_portal, -3, 100),
    "cc": (rd.cc, -100, 100),
    "cc1": (rd.cc1, -100, 100),
    "u_right": (rd.u_right, rd.u_right - 10, rd.u_right + 10),
    "rho_right": (rd.rho_right, rd.rho_right - 10, rd.rho_right + 10),
}

params = mymodel.make_params()
for name, (val, min_val, max_val) in param_settings.items():
    params.add(name=name, value=val, min=min_val, max=max_val)

# Fit the model
start_time = time.time()
result = mymodel.fit(exp_y_inp, params, x=np.arange(0, rd.total_time, rd.dt), nan_policy='omit')
fit_duration = time.time() - start_time

print('Optimization Completed in {:.2f} seconds'.format(fit_duration))
logging.info('Optimization Completed in %.2f seconds', fit_duration)

print_tuning_variables()

# Run the solver with optimized variables
start_time = time.time()
p_history = solver.solver(rd, logging)
spio.savemat('output/p_history_a.mat', {
    "p_history": p_history - rd.pressure_right,
    "x_probe": rd.x_probe,
    "label": "output",
    "t": np.arange(0, rd.total_time, rd.dt),
})
logging.info("Program Completed in %.2f seconds", time.time() - start_time)
print("Program Completed")

