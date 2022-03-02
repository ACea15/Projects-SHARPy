# Code to compare run the parametric hale parameters with static and modal solutions


import numpy as np
import os
import pdb
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra

importlib.reload(gm)
import sys

try:
    model_route = os.path.dirname(os.path.realpath(__file__)) + '/compare_hale'
except:
    model_route = os.getcwd() + '/aeroelasticPMOR_Optimization/parametric_aircraft/' + '/compare_hale'


def comp_settings(components=['fuselage', 'wing_r', 'winglet_r',
                              'wing_l', 'winglet_l', 'vertical_tail',
                              'horizontal_tail_right', 'horizontal_tail_left'],
                  bound_panels=8):
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
    g1c['fuselage'] = {'workflow': ['read_structure', 'read_aero'],
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
                                  'sweep': 0. * np.pi / 180,
                                  'dihedral': 0.},
                     'fem': {'stiffness_db': stiffness,
                             'mass_db': mass_wing,
                             'frame_of_reference_delta': [-1, 0., 0.]},
                     'aero': {'chord': [1., 1.],
                              'elastic_axis': 0.33,
                              'surface_m': bound_panels}
                     }
    g1c['winglet_r'] = {'workflow': ['create_structure', 'create_aero'],
                        'geometry': {'length': 4,
                                     'num_node': 3,
                                     'direction': [0., 1., 0.],
                                     'sweep': 0. * np.pi / 180,
                                     'dihedral': 20. * np.pi / 180},
                        'fem': {'stiffness_db': stiffness,
                                'mass_db': mass_wing,
                                'frame_of_reference_delta': [-1, 0., 0.]},
                        'aero': {'chord': [1., 1.],
                                 'elastic_axis': 0.33,
                                 'surface_m': bound_panels,
                                 'merge_surface': True}
                        }
    g1c['wing_l'] = {'symmetric': {'component': 'wing_r'}}
    g1c['winglet_l'] = {'symmetric': {'component': 'winglet_r'}}
    g1c['vertical_tail'] = {'workflow': ['create_structure', 'create_aero'],
                            'geometry': {'length': 2.5,
                                         'num_node': 11,
                                         'direction': [0., 0., 1.],
                                         'sweep': None,
                                         'dihedral': None},
                            'fem': {'stiffness_db': stiffness * sigma_tail,  # input tail stiffness
                                    'mass_db': mass_tail,
                                    'frame_of_reference_delta': [-1., 0., 0.]},
                            'aero': {'chord': [0.5, 0.5],
                                     'elastic_axis': 0.5,
                                     'surface_m': bound_panels}
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
    for ki in ['fuselage', 'wing_r', 'winglet_r',
               'wing_l', 'winglet_l', 'vertical_tail',
               'horizontal_tail_right', 'horizontal_tail_left']:

        if (ki not in ['include_aero', 'default_settings'] and
                ki not in components):
            del g1mm['assembly'][ki]

    return g1mm


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
# Modal solution                            #
#############################################
u_inf = 10
rho = 1.2
c_ref = 1.0
AoA = 0. * np.pi / 180
bound_panels = 8
sol_132 = {'sharpy': {'simulation_input': None,
                      'default_module': 'sharpy.routines.modal',
                      'default_solution': 'sol_132',
                      'default_solution_vars': {'num_modes': 10,
                                                'u_inf': u_inf,
                                                'rho': rho,
                                                'dt': c_ref / bound_panels / u_inf,
                                                'rotationA': [0., AoA, 0.],
                                                'panels_wake': 80,
                                                'horseshoe': True,
                                                'gravity_on': 0,
                                                'print_modal_matrices': False,
                                                'max_modal_disp': 0.15,
                                                'max_modal_rot_deg': 15.,
                                                'fsi_maxiter': 100,
                                                'fsi_tolerance': 1e-5,
                                                'fsi_relaxation': 0.1,
                                                'fsi_load_steps': 20,
                                                's_maxiter': 100,
                                                's_tolerance': 1e-5,
                                                's_relaxation': 1e-3,
                                                's_load_steps': 1,
                                                's_delta_curved': 1e-4,
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
AoA = 4.072 * np.pi / 180
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
                   'fsi_load_steps': 1,
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
#####################################################################
# Run a flutter solution around an arbitrary aeroelastic equilibrium#
####################################################################
u_inf = 20
rho = 1.2
c_ref = 1.0
AoA = 0. * np.pi / 180
bound_panels = 8
sol_152 = {'sharpy': {'simulation_input': None,
                      'default_module': 'sharpy.routines.flutter',
                      'default_solution': 'sol_152',
                      'default_solution_vars': {
                          'flutter_reference': 21.,
                          'root_method': 'bisection',
                          'velocity_increment': 10.,
                          'flutter_error': 0.001,
                          'damping_tolerance': 5e-3,
                          'inout_coordinates': 'modes',
                          'secant_max_calls': 15,
                          'rho': rho,
                          'gravity_on': False,
                          'u_inf': u_inf,
                          'panels_wake': bound_panels * 10,
                          'dt': c_ref / bound_panels / u_inf,
                          'c_ref': c_ref,
                          'rom_method': '',
                          'rotationA': [0., AoA, 0.],
                          'horseshoe': True,
                          'num_modes': 20,
                          'fsi_maxiter': 100,
                          'fsi_tolerance': 1e-5,
                          'fsi_relaxation': 0.3,
                          'fsi_load_steps': 1,
                          's_maxiter': 100,
                          's_tolerance': 1e-5,
                          's_relaxation': 1e-3,
                          's_load_steps': 1,
                          's_delta_curved': 1e-4,
                          'add2_flow': [['StaticCoupled', 'plot']],
                      },
                      'default_sharpy': {},
                      'model_route': None}}
#############################################


solutions = dict()  # dictionary with solutions mapping
solutions['0'] = sol_0
solutions['112'] = sol_112
solutions['132'] = sol_132
solutions['152'] = sol_152

sol_i = '132'  # pick solution to run
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
              model_dict=model_settings('test_%s' % sol_i),

              components_dict=comp_settings(bound_panels=bound_panels),
              simulation_dict=solutions[sol_i])
#############Create another model by reading the .h5 files directly


data = g1.run()