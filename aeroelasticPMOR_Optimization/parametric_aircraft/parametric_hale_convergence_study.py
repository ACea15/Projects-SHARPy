# Code for convergence study of Hale
# Date: 18/02/22
# Author: Pablo de Felipe
import numpy as np
import os
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra

importlib.reload(gm)
import sys

try:
    model_route = os.path.dirname(os.path.realpath(__file__)) + '/single_test_run'
except:
    model_route = os.getcwd() + '/aeroelasticPMOR_Optimization/parametric_aircraft/' + '/single_test_run'
# aeroelasticity parameters
main_ea = 0.3  # Wing elastic axis from LE as %
main_cg = 0.3  # Not sure about this input
sigma = 1

# other
c_ref = 1.0
# Wing Stiffness & mass
ea, ga = 1.5e7, 1e5
gj = 1.5e4
eiy = 3e4
eiz = 6e5
base_stiffness = np.diag([ea, ga, ga, sigma * gj, sigma * eiy, eiz])
stiffness = np.zeros((1, 6, 6))
stiffness[0] = base_stiffness
m_unit = 0.75
j_tors = 0.075
pos_cg_b = np.array([0., c_ref * (main_cg - main_ea), 0.])
m_chi_cg = algebra.skew(m_unit * pos_cg_b)
mass_wing = np.zeros((1, 6, 6))
mass_wing[0, :, :] = np.diag([m_unit, m_unit, m_unit,
                              j_tors, .5 * j_tors, .5 * j_tors])
mass_wing[0, :3, 3:] = m_chi_cg
mass_wing[0, 3:, :3] = -m_chi_cg

# Tail Stiffness and mass of the horizontal tail
ea_tail = 0.5
sigma_tail = 10  # Use a multiplication factor
m_unit_tail = 0.3
j_tors_tail = 0.08

mass_tail = np.zeros((1, 6, 6))
mass_tail[0, :, :] = np.diag([m_unit_tail,
                              m_unit_tail,
                              m_unit_tail,
                              j_tors_tail,
                              .5 * j_tors_tail,
                              .5 * j_tors_tail])
mass_tail[0, :3, 3:] = m_chi_cg
mass_tail[0, 3:, :3] = -m_chi_cg
# Fuselage Stiffness and mass
sigma_fuselage = 10
m_unit_fuselage = 0.2
j_tors_fuselage = 0.08
mass_fuselage = np.zeros((1, 6, 6))
mass_fuselage[0, :, :] = np.diag([m_unit_fuselage,
                                  m_unit_fuselage,
                                  m_unit_fuselage,
                                  j_tors_fuselage,
                                  .5 * j_tors_fuselage,
                                  .5 * j_tors_fuselage])
mass_fuselage[0, :3, 3:] = m_chi_cg
mass_fuselage[0, 3:, :3] = -m_chi_cg

# Lumped mass
n_lumped_mass = 1  # Number of lumped masses
lumped_mass_nodes = np.zeros((n_lumped_mass,), dtype=int)  # Maps lumped mass to nodes
lumped_mass = np.zeros((n_lumped_mass,))  # Array of lumped masses in kg
lumped_mass[0] = 50
lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))  # 3x3 inertia to the previous masses
lumped_mass_position = np.zeros((n_lumped_mass, 3))  # Relative position to the belonging node in B FoR

g1c = dict()
g1c['fuselage'] = {'workflow': ['create_structure', 'create_aero0'],
                   'geometry': {'length': 10,
                                'num_node': 11,
                                'direction': [1., 0., 0.],
                                'sweep': 0.,
                                'dihedral': 0.},
                   'fem': {'stiffness_db': stiffness,
                           'mass_db': mass_fuselage,
                           'frame_of_reference_delta': [0, 1., 0.],
                           'lumped_mass': lumped_mass,
                           'lumped_mass_nodes': lumped_mass_nodes,
                           'lumped_mass_inertia': lumped_mass_inertia,
                           'lumped_mass_position': lumped_mass_position}
                   }

g1c['wing_r'] = {'workflow': ['create_structure', 'create_aero'],
                 'geometry': {'length': 20.,
                              'num_node': 11,
                              'direction': [0., 1., 0.],
                              'sweep': 20. * np.pi / 180,
                              'dihedral': 0.},
                 'fem': {'stiffness_db': stiffness,
                         'mass_db': mass_wing,
                         'frame_of_reference_delta': [-1, 0., 0.]},
                 'aero': {'chord': [1., 1.],
                          'elastic_axis': 0.33,
                          'surface_m': 16}
                 }
g1c['wing_l'] = {'symmetric': {'component': 'wing_r'}}
g1c['winglet_r'] = {'workflow': ['create_structure', 'create_aero'],
                    'geometry': {'length': 4,
                                 'num_node': 3,
                                 'direction': [0., 1., 0.],
                                 'sweep': 20. * np.pi / 180,
                                 'dihedral': 20. * np.pi / 180},
                    'fem': {'stiffness_db': stiffness,
                            'mass_db': mass_wing,
                            'frame_of_reference_delta': [-1, 0., 0.]},
                    'aero': {'chord': [1., 1.],
                             'elastic_axis': 0.33,
                             'surface_m': 16}
                    }
g1c['winglet_l'] = {'symmetric': {'component': 'winglet_r'}}

g1c['vertical_tail'] = {'workflow': ['create_structure', 'create_aero'],
                        'geometry': {'length': 2.5,
                                     'num_node': 11,
                                     'direction': [0., 0., 1.],
                                     'sweep': None,
                                     'dihedral': None},
                        'fem': {'stiffness_db': stiffness * sigma_tail,  # input tail stiffness
                                'mass_db': mass_tail,
                                'frame_of_reference_delta': [-1, 0., 0.]},
                        'aero': {'chord': [0.5, 0.5],
                                 'elastic_axis': 0.5,
                                 'surface_m': 16}
                        }
g1c['horizontal_tail_right'] = {'workflow': ['create_structure', 'create_aero'],
                                'geometry': {'length': 2.5,
                                             'num_node': 11,
                                             'direction': [0., 1., 0.],
                                             'sweep': 0.,
                                             'dihedral': 0.},
                                'fem': {'stiffness_db': stiffness * sigma_tail,
                                        'mass_db': mass_tail,
                                        'frame_of_reference_delta': [-1, 0., 0.]},
                                'aero': {'chord': [0.5, 0.5],
                                         'elastic_axis': 0.4,
                                         'surface_m': 16}
                                }
g1c['horizontal_tail_left'] = {'symmetric': {'component': 'horizontal_tail_right'}}

g1mm = {'model_name': 'hale_convergence',
        'model_route': model_route,
        'iterate_type': 'Full_Factorial',
        'write_iterate_vars': True,
        'iterate_vars': {},
        'iterate_labels': {'label_type': 'number',
                           'print_name_var': 0},
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
                                   'node_in_upstream': 10},
                     'wing_l': {'upstream_component': 'fuselage',
                                'node_in_upstream': 0},
                     'winglet_l': {'upstream_component': 'wing_l',
                                   'node_in_upstream': 10},
                     'vertical_tail': {'upstream_component': 'fuselage',
                                       'node_in_upstream': 10},
                     'horizontal_tail_right': {'upstream_component': 'vertical_tail',
                                               'node_in_upstream': 10},
                     'horizontal_tail_left': {'upstream_component': 'vertical_tail',
                                              'node_in_upstream': 10}
                     }
        }

g1sm = {'sharpy': {'simulation_input': None,
                   'default_module': 'sharpy.routines.flutter',
                   'default_solution': 'sol_146',
                   'default_solution_vars': {'num_modes': 6,
                                             'alpha': 1.,
                                             'beta': .0,
                                             'roll': .0,
                                             'panels_wake': 16 * 5,
                                             'AerogridPlot': {'include_rbm': 'on',
                                                              'include_applied_forces': 'on',
                                                              'minus_m_star': 0},
                                             'BeamPlot': {'include_rbm': 'off',
                                                          'include_applied_forces': 'off'},
                                             'rom_method':''},
                                             'default_sharpy': {},
                                             'model_route': None}}

g1 = gm.Model('sharpy', ['sharpy'], model_dict=g1mm, components_dict=g1c,
              simulation_dict=g1sm)
# g1.build()
data = g1.run()

