# Code for convergence study of Hale
# Date: 18/02/22
# Author: Pablo de Felipe
import numpy as np
import os
import pdb
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra

importlib.reload(gm)
import sys

try:
    model_route = os.path.dirname(os.path.realpath(__file__)) + '/simplehale_verification'
except:
    import inspect
    __file__ = inspect.getfile(lambda: None)
    model_route = os.path.dirname(__file__) + '/simplehale_verification'

def comp_settings():


    ##############
    # Components #
    ##############
    g1c = dict()
    g1c['hale'] = {'workflow': ['read_structure', 'read_aero'],
                   'read_structure_file': os.path.dirname(__file__) + \
                   'simple_HALE.fem.h5',
                   'read_aero_file': os.path.dirname(__file__) + \
                   'simple_HALE.aero.h5'
                       }

    return g1c

def model_settings(model_name):
    g1mm = {'model_name': model_name,
            'model_route': model_route
            }

    return g1mm
    
##############################################
# Plot the initial model
##############################################

bound_panels = 8
sol_0 = {'sharpy': {'simulation_input': None,
                   'default_module': 'sharpy.routines.basic',
                   'default_solution': 'sol_0',
                   'default_solution_vars': {'panels_wake':  bound_panels * 5,
                                             'add2_flow': \
                                             [['AerogridLoader','WriteVariablesTime']],
                                             'WriteVariablesTime': \
                                             {'structure_variables':
                                              ['pos', 'psi'],
                                              'structure_nodes':list(range(20)),
                                              'cleanup_old_solution': 'on'}},
                   'default_sharpy': {},
                   'model_route': None}}
#############################################
# Modal solution                            # 
#############################################
u_inf = 20
rho = 1.2
c_ref = 1.0
AoA = 0.*np.pi/180
bound_panels = 8
sol_132 = {'sharpy': {'simulation_input': None,
           'default_module': 'sharpy.routines.modal',
           'default_solution': 'sol_132',
           'default_solution_vars': {'num_modes': 10,
                                     'u_inf': u_inf,
                                     'rho': rho,
                                     'dt': c_ref / bound_panels / u_inf,
                                     'rotationA':[0., AoA, 0.],
                                     'panels_wake':1,         
                                     'horseshoe': True,       
                                     'gravity_on':0,
                                     'print_modal_matrices':False,  
                                     'max_modal_disp':0.15,
                                     'max_modal_rot_deg':15.,
                                     'fsi_maxiter':100,       
                                     'fsi_tolerance':1e-5,    
                                     'fsi_relaxation':0.1,   
                                     'fsi_load_steps':20,      
                                     's_maxiter':100,         
                                     's_tolerance':1e-5,      
                                     's_relaxation':1e-3,     
                                     's_load_steps':1,        
                                     's_delta_curved':1e-4,   
           },
           'default_sharpy': {},
           'model_route': None
           }
         }
#############################################
#  Aeroelastic equilibrium                  #
#############################################
u_inf = 10
rho = 1.2
c_ref = 1.0
AoA = 2*np.pi/180
bound_panels = 8
sol_112 = {
    'sharpy': {'simulation_input': None,
               'default_module': 'sharpy.routines.static',
               'default_solution': 'sol_112',
               'default_solution_vars': {
                   'u_inf': u_inf,
                   'rho': rho,
                   'gravity_on': False,
                   'dt': c_ref / bound_panels / u_inf,
                   'panels_wake': bound_panels * 5,
                   'rotationA': [0., AoA, 0.],
                   'horseshoe': False,
                   'fsi_maxiter': 100,    
                   'fsi_tolerance': 1e-5, 
                   'fsi_relaxation': 0.1,
                   'fsi_load_steps': 5,  
                   's_maxiter': 100,      
                   's_tolerance': 1e-5,   
                   's_relaxation': 1e-3, 
                   's_load_steps': 1,     
                   's_delta_curved': 1e-4,
                   'add2_flow': [['StaticCoupled', ['plot', 'AeroForcesCalculator']]],
                   'AeroForcesCalculator': {'write_text_file':True},
                   # 'u_inf_direction': [np.cos(deg_to_rad(3.)),
                   #                     0., np.sin(deg_to_rad(3.))]
                   },
               'default_sharpy': {},
               'model_route': None
               }
}
#####################################################################
# Run a flutter solution around an arbitrary aeroelastic equilibrium#
####################################################################
u_inf = 10
rho = 1.2
c_ref = 1.0
AoA = 0.*np.pi/180
bound_panels = 8
sol_152 = {'sharpy': {'simulation_input': None,
               'default_module': 'sharpy.routines.flutter',
               'default_solution': 'sol_152',
               'default_solution_vars': {
                   'flutter_reference': 21.,
                   'root_method':'bisection',
                   'velocity_increment': 10.,
                   'flutter_error': 0.001,
                   'damping_tolerance': 0.,
                   'inout_coordinates': 'modes',
                   'secant_max_calls':15,
                   'rho': rho,
                   'gravity_on': False,
                   'u_inf': u_inf,                           
                   'panels_wake': bound_panels * 10,
                   'dt': c_ref / bound_panels / u_inf,         
                   'c_ref': c_ref,
                   'rom_method': '',                           
                   'rotationA':[0., AoA, 0.],
                   'horseshoe': True,                           
                   'num_modes': 20,
                   'fsi_maxiter':100,       
                   'fsi_tolerance':1e-5,    
                   'fsi_relaxation':0.3,   
                   'fsi_load_steps':1,      
                   's_maxiter':100,         
                   's_tolerance':1e-5,      
                   's_relaxation':1e-3,     
                   's_load_steps':1,        
                   's_delta_curved':1e-4,
                   'add2_flow': [['StaticCoupled', 'plot']],
               },
               'default_sharpy': {},
               'model_route': None}}
#############################################



solutions = dict() # dictionary with solutions mapping
solutions['0'] = sol_0
solutions['112'] = sol_112
solutions['132'] = sol_132
solutions['152'] = sol_152

sol_i = '152' # pick solution to run
####### choose components to analyse ######### 
# g1 = gm.Model('sharpy', ['sharpy'],
#               model_dict=model_settings('test_%s'%sol_i,
#                                         ['fuselage','wing_r','winglet_r',
#                                          'wing_l','winglet_l']),
#               components_dict=comp_settings(['fuselage','wing_r','winglet_r',
#                                              'wing_l','winglet_l']),
#               simulation_dict=solutions[sol_i])
####### ... or do full aircraft #########
g1 = gm.Model('sharpy', ['sharpy'],
              model_dict=model_settings('test_%s'%sol_i),
              components_dict=comp_settings(),
              simulation_dict=solutions[sol_i])

data = g1.run()

