""" Debug module used for testing eqv methods"""

import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
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