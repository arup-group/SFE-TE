import heating_regimes as hr
import heat_transfer_models as htm
import equivalent_curves as ecr
import risk_models as rm

UNIT_CATALOGUE = {
    'A_c': {'ui_label': 'ca', 'title': 'Compartment area', 'unit': 'm$^2$'},
    'c_ratio': {'ui_label': 'csr', 'title': 'Compartment sides ratio', 'unit': '-'},
    'h_c': {'ui_label': 'ch', 'title': 'Compartment height', 'unit': 'm'},
    'w_frac': {'ui_label': 'vpr', 'title': 'Ventilated perimeter fraction', 'unit': '-'},
    'h_w_eq': {'ui_label': 'heq', 'title': 'Average window height', 'unit': 'm'},
    'remain_frac': {'ui_label': 'trf', 'title': 'Thermal resilience', 'unit': '-'},
    'fabr_inrt': {'ui_label': 'ftr', 'title': 'Fabric thermal inertia', 'unit': 'J/m$^2$s$^{1/2}$K'},
    'q_f_d': {'ui_label': 'fl', 'title': 'Fuel load', 'unit': 'MJ/m$^2$'},
    'Q': {'ui_label': 'fl', 'title': 'Heat release rate per unit area', 'unit': 'KW/m$^2$'},
    't_lim': {'ui_label': 'fl', 'title': 'Fire growth rate', 'unit': 'min'},
    'spr_rate': {'ui_label': 'fl', 'title': 'Fire spread rate', 'unit': 'mm/s'},
    'flap_angle': {'ui_label': 'fl', 'title': 'Flapping angle', 'unit': 'deg'},
    'T_nf_max': {'ui_label': 'fl', 'title': 'Near field max temperature', 'unit': '°C'}}

HEATING_REGIMES = {
        'Uniform BS EN 1991-1-2': [hr.UniEC1, {}],
        'Traveling ISO 16733-2': [hr.TravelingISO16733, {}]}

EQV_METHODS = {
    '0d_ht_en_1993_1_2':  [htm.SteelEC3, {}]}

EQV_CURVES = {
    'ISO_834': [ecr.StandardFire, {}]}

RISK_METHODS = {
    'bs_9999': [rm.Kirby, {}]}