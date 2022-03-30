# Code to obtain wing deformations i.e deflection at the tip and twist at some other station
# Date of start: 11/03/22
# Author: Pablo de Felipe

import numpy as np
import os
import pdb
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra
import pandas as pd

import sharpy.utils.generate_cases as gc
import sharpy.utils.h5utils as h5utils
import sharpy.utils.solver_interface as solver_interface
import sharpy.sharpy_main

importlib.reload(gm)
import sys

try:
    model_route = os.path.dirname(os.path.realpath(__file__)) + '/aspectRatio_wingDef'
except:
    import inspect

    __file__ = inspect.getfile(lambda: None)
    model_route = os.path.dirname(__file__) + '/aspectRatio_wingDef'


def comp_settings(wing_semispan,
                  components=['fuselage', 'wing_r', 'winglet_r',
                              'wing_l', 'winglet_l', 'vertical_tail',
                              'horizontal_tail_right', 'horizontal_tail_left'],
                  bound_panels=8):
    # aeroelasticity parameters
    main_ea = 0.3  # Wing elastic axis from LE as %
    main_cg = 0.3  # Not sure about this input
    sigma = 1.5
    c_ref = 1.0

    #########
    # wings #
    #########
    #
    ea = 1e7
    ga = 1e5
    gj = 1e4
    eiy = 2e4
    eiz = 4e6
    m_bar_main = 0.75
    j_bar_main = 0.075
    mass_main1 = np.diag([m_bar_main, m_bar_main, m_bar_main,
                          j_bar_main, 0.5 * j_bar_main, 0.5 * j_bar_main])
    stiffness_main1 = sigma * np.diag([ea, ga, ga, gj, eiy, eiz])
    stiffness_main = np.zeros((1, 6, 6))
    stiffness_main[0] = stiffness_main1
    mass_main = np.zeros((1, 6, 6))
    mass_main[0] = mass_main1
    ############
    # fuselage #
    ############
    #
    sigma_fuselage = 10
    m_bar_fuselage = 0.2
    j_bar_fuselage = 0.08
    stiffness_fuselage1 = np.diag([ea, ga, ga, gj, eiy, eiz]) * sigma * sigma_fuselage
    stiffness_fuselage1[4, 4] = stiffness_fuselage1[5, 5]
    mass_fuselage1 = np.diag([m_bar_fuselage,
                              m_bar_fuselage,
                              m_bar_fuselage,
                              j_bar_fuselage,
                              j_bar_fuselage * 0.5,
                              j_bar_fuselage * 0.5])
    stiffness_fuselage = np.zeros((1, 6, 6))
    stiffness_fuselage[0] = stiffness_fuselage1
    mass_fuselage = np.zeros((1, 6, 6))
    mass_fuselage[0] = mass_fuselage1
    ########
    # tail #
    ########
    #
    sigma_tail = 100
    m_bar_tail = 0.3
    j_bar_tail = 0.08
    stiffness_tail1 = np.diag([ea, ga, ga, gj, eiy, eiz]) * sigma * sigma_tail
    stiffness_tail1[4, 4] = stiffness_tail1[5, 5]
    mass_tail1 = np.diag([m_bar_tail,
                          m_bar_tail,
                          m_bar_tail,
                          j_bar_tail,
                          j_bar_tail * 0.5,
                          j_bar_tail * 0.5])
    stiffness_tail = np.zeros((1, 6, 6))
    stiffness_tail[0] = stiffness_tail1
    mass_tail = np.zeros((1, 6, 6))
    mass_tail[0] = mass_tail1

    ######################################
    # Lumped mass at fuselage/wing cross #
    ######################################
    n_lumped_mass = 1  # Number of lumped masses
    lumped_mass_nodes = np.zeros((n_lumped_mass,), dtype=int)  # Maps lumped mass to nodes
    lumped_mass = np.zeros((n_lumped_mass,))  # Array of lumped masses in kg
    lumped_mass[0] = 50
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))  # 3x3 inertia to the previous masses
    lumped_mass_position = np.zeros((n_lumped_mass, 3))  # Relative position to the belonging node in B FoR

    ##############
    # Components #
    ##############
    g1c = dict()
    g1c['fuselage'] = {'workflow': ['create_structure', 'create_aero0'],
                       'geometry': {'length': 10,
                                    'num_node': 9,
                                    'direction': [1., 0., 0.],
                                    'sweep': 0.,
                                    'dihedral': 0.},
                       'fem': {'stiffness_db': stiffness_fuselage,
                               'mass_db': mass_fuselage,
                               'frame_of_reference_delta': [0, 1., 0.],
                               'lumped_mass': lumped_mass,
                               'lumped_mass_nodes': lumped_mass_nodes,
                               'lumped_mass_inertia': lumped_mass_inertia,
                               'lumped_mass_position': lumped_mass_position}
                       }

    g1c['wing_r'] = {'workflow': ['create_structure', 'create_aero'],
                     'geometry': {'length': wing_semispan,
                                  'num_node': 13,
                                  'direction': [0., 1., 0.],
                                  'sweep': 0. * np.pi / 180,
                                  'dihedral': 0.},
                     'fem': {'stiffness_db': stiffness_main,
                             'mass_db': mass_main,
                             'frame_of_reference_delta': [-1, 0., 0.]},
                     'aero': {'chord': [1., 1.],
                              'elastic_axis': main_ea,
                              'surface_m': bound_panels}
                     }
    g1c['winglet_r'] = {'workflow': ['create_structure', 'create_aero'],
                        'geometry': {'length': 4,
                                     'num_node': 5,
                                     'direction': [0., 1., 0.],
                                     'sweep': 0. * np.pi / 180,
                                     'dihedral': 20. * np.pi / 180},
                        'fem': {'stiffness_db': stiffness_main,
                                'mass_db': mass_main,
                                'frame_of_reference_delta': [-1, 0., 0.]},
                        'aero': {'chord': [1., 1.],
                                 'elastic_axis': main_ea,
                                 'surface_m': bound_panels,
                                 'merge_surface': True}
                        }
    g1c['wing_l'] = {'symmetric': {'component': 'wing_r'}}
    g1c['winglet_l'] = {'symmetric': {'component': 'winglet_r'}}
    g1c['vertical_tail'] = {'workflow': ['create_structure', 'create_aero'],
                            'geometry': {'length': 2.5,
                                         'num_node': 9,
                                         'direction': [0., 0., 1.],
                                         'sweep': None,
                                         'dihedral': None},
                            'fem': {'stiffness_db': stiffness_tail,
                                    'mass_db': mass_tail,
                                    'frame_of_reference_delta': [-1., 0., 0.]},
                            'aero': {'chord': [0.45, 0.45],
                                     'elastic_axis': 0.5,
                                     'surface_m': bound_panels}
                            }
    g1c['horizontal_tail_right'] = {'workflow': ['create_structure', 'create_aero'],
                                    'geometry': {'length': 2.5,
                                                 'num_node': 9,
                                                 'direction': [0., 1., 0.],
                                                 'sweep': 0.,
                                                 'dihedral': 0.},
                                    'fem': {'stiffness_db': stiffness_tail,
                                            'mass_db': mass_tail,
                                            'frame_of_reference_delta': [-1, 0., 0.]},
                                    'aero': {'chord': [0.5, 0.5],
                                             'elastic_axis': 0.5,
                                             'surface_m': bound_panels}
                                    }
    g1c['horizontal_tail_left'] = {'symmetric': {'component': 'horizontal_tail_right'}}

    g1c_output = {i: g1c[i] for i in components}
    return g1c_output


def model_settings(model_name,
                   components=['fuselage', 'wing_r', 'winglet_r',
                               'wing_l', 'winglet_l', 'vertical_tail',
                               'horizontal_tail_right', 'horizontal_tail_left']):
    g1mm = {'model_name': model_name,
            'model_route': model_route,
            # 'iterate_type': 'Full_Factorial',
            # 'write_iterate_vars': True,
            # 'iterate_vars': {'fuselage*geometry-length': np.linspace(7, 15., 3),
            #                  'wing_r*geometry-length': np.linspace(15, 25., 3),
            #                  'winglet_r*geometry-dihedral': np.pi / 180 * np.array([0, 20, 40])},
            # 'iterate_labels': {'label_type': 'number',
            #                    'print_name_var': 0},
            'assembly': {'include_aero': 1,
                         'default_settings': 1,  # beam_number and aero surface and
                         # surface_distribution
                         # selected by default one
                         # per component
                         'fuselage': {'upstream_component': '',
                                      'node_in_upstream': 0},
                         'wing_r': {'keep_aero_node': 1,
                                    'upstream_component': 'fuselage',
                                    'node_in_upstream': 0},
                         'winglet_r': {'keep_aero_node': 1,
                                       'upstream_component': 'wing_r',
                                       'node_in_upstream': -1},
                         'wing_l': {'upstream_component': 'fuselage',
                                    'node_in_upstream': 0},
                         'winglet_l': {'upstream_component': 'wing_l',
                                       'node_in_upstream': -1},
                         'vertical_tail': {'upstream_component': 'fuselage',
                                           'node_in_upstream': -1},
                         'horizontal_tail_right': {'upstream_component': 'vertical_tail',
                                                   'node_in_upstream': -1},
                         'horizontal_tail_left': {'upstream_component': 'vertical_tail',
                                                  'node_in_upstream': -1}
                         }
            }
    for ki in ['fuselage', 'wing_r', 'winglet_r',
               'wing_l', 'winglet_l', 'vertical_tail',
               'horizontal_tail_right', 'horizontal_tail_left']:

        if (ki not in ['include_aero', 'default_settings'] and
                ki not in components):
            del g1mm['assembly'][ki]

    return g1mm

# Calculate the wing semispan for different aspect ratios
#AR = np.array((10,20,30,40))
#AR = np.array((10,20))
AR = np.array((20,25,35))
winglt_length = 4.0
winglt_dhdrl  = 20*np.pi/180
wing_chord    = 1.0
wing_semispan = 0.5*AR*wing_chord-winglt_length*np.cos(winglt_dhdrl)
##############################################
# Plot the initial model
##############################################

bound_panels = 8
sol_0 = {'sharpy': {'simulation_input': None,
                    'default_module': 'sharpy.routines.basic',
                    'default_solution': 'sol_0',
                    'default_solution_vars': {'panels_wake': bound_panels * 5,
                                              'add2_flow': \
                                                  [['AerogridLoader', 'WriteVariablesTime']],
                                              'WriteVariablesTime': \
                                                  {'structure_variables':
                                                       ['pos', 'psi'],
                                                   'structure_nodes': list(range(20)),
                                                   'cleanup_old_solution': 'on'}},
                    'default_sharpy': {},
                    'model_route': None}}

#############################################
#  Aeroelastic equilibrium                  #
#############################################
u_inf = 10
rho = 1.2
c_ref = 1.0
AoA_deg = 3.5
AoA = AoA_deg * np.pi / 180
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
                   'AeroForcesCalculator': {'write_text_file': True},
                   # 'u_inf_direction': [np.cos(deg_to_rad(3.)),
                   #                     0., np.sin(deg_to_rad(3.))]
               },
               'default_sharpy': {},
               'model_route': None
               }
}
# Preallocate arrays to store in data
print(len(AR))

wing_def = np.zeros((len(AR),))
psi      = wing_def
sol_i = '112'  # pick solution to run
####### choose components to analyse #########
# g1 = gm.Model('sharpy', ['sharpy'],
#               model_dict=model_settings('test_%s'%sol_i,
#                                         ['fuselage','wing_r','winglet_r',
#                                          'wing_l','winglet_l']),
#               components_dict=comp_settings(['fuselage','wing_r','winglet_r',
#                                              'wing_l','winglet_l']),
#               simulation_dict=solutions[sol_i])
####### ... or do full aircraft #########
for i in range(len(AR)):
    g1 = gm.Model('sharpy', ['sharpy'],
                  model_dict=model_settings('ar_%s' %AR[i]),
                  components_dict=comp_settings(wing_semispan[i],
                                                bound_panels=bound_panels),
                  simulation_dict=sol_112)

    #data = g1.run()
    # For some reason data is empty, have tried data[0] in notebook but does not help
    # As a fix run it manually
    os.chdir('./aspectRatio_wingDef/ar_'+'%s' %AR[i])
    data = sharpy.sharpy_main.main(['', 'ar_%s.sharpy' %AR[i]])
    # Store the values from the data file

    yc_ini = data.structure.fortran['pos_ini'][:, 1] # Undeformed y coordinates
    print(yc_ini)
    print(wing_semispan[i])
    print(np.where(yc_ini == wing_semispan[i]))
    #wing_node = int(np.where(yc_ini == wing_semispan[i])[0][0])
    wing_node  = 20
    coordinates = data.structure.timestep_info[0].pos
    #rotations   = data.structure.timestep_info[0].psi
    xc = coordinates[:, 0]
    yc = coordinates[:, 1]
    zc = coordinates[:, 2]

    wing_def[i] = zc[wing_node]
    # Go back to the surrogate directory
    os.chdir('/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/surrogate_model/')

# Export results via a pandas DataFrame
data = {
    "AR" : AR,
    "wing_Def": wing_def,
    "semi_span": wing_semispan
}
# Create the pandas data frame
data_pandas = pd.DataFrame(data)
# Change the directory to save in the model route folder
os.chdir(model_route)
# Write to a csv file
data_pandas.to_csv('ar_wing_def_testing2.csv')





