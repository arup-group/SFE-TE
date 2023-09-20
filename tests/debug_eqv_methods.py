""" Debug module used for testing eqv methods"""

import sys
if '..' not in sys.path:
    sys.path.insert(0, '../src')
from src import equivalence_methods as em
from src import configs as cfg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

SteelEC3_configs = {

    'A_v': 70, # section factor
    'c_p': 900, # protection heat capacity
    'k_p': 0.17, # protection coductivity
    'ro_p': 700, # protection density
    'T_amb': 20, # ambient temp
    'dt': 30, #time step
    'lim_temp': 550,  # limiting temperature
    'eqv_max': 180,  # eqv max time
    'eqv_step': 10,
    'max_itr': 10,
    'tol': 5,
    'prot_thick_range': [0.00025, 0.1, 0.0005]}

model = em.SteelEC3(equivalent_curve='iso_834', **SteelEC3_configs)
model.get_equivalent_protection()
data = pd.DataFrame(model.equiv_prot_req_data, columns=['prot_thickness', 'exposure'])
# plt.plot(model.equiv_prot_req_data[:,1], model.equiv_prot_req_data[:,0]*1000)
# plt.show()

### DEBUGGING thermal response methods
def exp_fxn(t, subsample_mask):
    result = np.array([0.0, 0.0])

    if t <= 60:
        result[0] = 1000
        result[1] = 20 + 980 * t / 60
    elif t <= 120:
        result[0] = 1000
        result[1] = 1000 - 980 * (t - 60) / 60
    else:
        result[0] = 1000
        result[1] = 20

    return result

inputs = {
        "equivalent_curve": "iso_834",
        "A_v": 100,
        "c_p": 900,
        "k_p": 0.17,
        "ro_p": 700,
        "T_amb": 20,
        "dt": 30,
        "lim_temp": 550,
        "eqv_max": 180,
        "eqv_step": 5,
        'max_itr': 10,
        'tol': 5,
        'prot_thick_range': [0.0001, 0.1, 0.0005]}

eq_method = eqm.SteelEC3(**inputs)
eq_method.get_equivalent_protection()

# test t = 15
temps = []
for k in np.arange(0, 15, 0.5):
    temps.append(exp_fxn(k,0))
T_max, all_temps = eq_method.calc_thermal_response(
    equiv_exp=90,
    exposure_fxn=exp_fxn,
    t_final=15,
    sample_size=2,
    output_history=True,
    early_stop=10000)