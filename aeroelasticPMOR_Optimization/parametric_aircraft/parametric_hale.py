import numpy as np
import os
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra
importlib.reload(gm)
import sys

split_path = [sys.path[i].split('/') for i in range(len(sys.path))]
for i in range(len(split_path)):
    if 'sharpy' in split_path[i]:
        ind = i
        break

sharpy_dir = sys.path[ind]
# aeroelasticity parameters
main_ea = 0.33
main_cg = 0.43
sigma = 1

# other
c_ref = 1.0

ea, ga = 1e9, 1e9
gj = 0.987581e6
eiy = 9.77221e6
eiz = 1e2 * eiy
base_stiffness = np.diag([ea, ga, ga, sigma * gj, sigma * eiy, eiz])
stiffness = np.zeros((1, 6, 6))
stiffness[0] = base_stiffness
m_unit = 35.71
j_tors = 8.64
pos_cg_b = np.array([0., c_ref * (main_cg - main_ea), 0.])
m_chi_cg = algebra.skew(m_unit * pos_cg_b)
mass = np.zeros((1, 6, 6))
mass[0, :, :] = np.diag([m_unit, m_unit, m_unit,
                         j_tors, .1 * j_tors, .9 * j_tors])
mass[0, :3, 3:] = m_chi_cg
mass[0, 3:, :3] = -m_chi_cg


g1c = dict()
g1c['fuselage'] = {'workflow':['create_structure','create_aero0'],
                  'geometry': {'length':10,
                               'num_node':11,
                               'direction':[1.,0.,0.],
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[0,1.,0.]}
}

g1c['wing_r'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':12,
                               'num_node':11,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[1.,1.],
                           'elastic_axis':0.33,
                           'surface_m':16}
                 }
g1c['wing_l'] = {'symmetric': {'component':'wing_r'}}
g1c['winglet_r']= {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':4,
                               'num_node':3,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':-20.*np.pi/180},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[1.,1.],
                           'elastic_axis':0.33,
                           'surface_m':16}
                 }
# Symmetric for winglets does not work
#g1c['winglet_l'] = {'symmetric': {'component':'winglet_r'}}
g1c['winglet_l']= {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':4,
                               'num_node':3,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':-20.*np.pi/180},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[1., 0., 0.]}, # Use [1.,0.,0.] to try and make it a left wing
                  'aero': {'chord':[1.,1.],
                           'elastic_axis':0.33,
                           'surface_m':16}
                 }

g1c['vertical_tail'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':2.5,
                               'num_node':11,
                               'direction':[0.,0.,1.],
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[0.5,0.5],
                           'elastic_axis':0.5,
                           'surface_m':16}
                 }
g1c['horizontal_tail_right'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':2.5,
                               'num_node':11,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[0.5,0.5],
                           'elastic_axis':0.4,
                           'surface_m':16}
                 }
g1c['horizontal_tail_left'] = {'symmetric': {'component':'horizontal_tail_right'}}

g1mm = {'model_name':'hale_test',
        'model_route':os.path.dirname(os.path.realpath(__file__)),
        'iterate_type': 'Full_Factorial',
        'iterate_vars': {'wing_r*geometry-sweep':np.pi/180*np.array([0,20,40]),
                         'wing_r*geometry-length':np.linspace(14,18.,3)},
        'iterate_labels': {'label_type':'number',
                           'print_name_var':0},
        'assembly': {'include_aero':1,
                     'default_settings': 1, # beam_number and aero surface and
                                            # surface_distribution
                                            # selected by default one
                                            # per component
                     'fuselage':{'node2add':0,
                                 'upstream_component':'',
                                 'node_in_upstream':0},
                     'wing_r':{'keep_aero_node':1,
                               'upstream_component':'fuselage',
                               'node_in_upstream':0},
                     'winglet_r':{'keep_aero_node':1,
                               'upstream_component':'wing_r',
                               'node_in_upstream':10},
                     'wing_l':{'upstream_component':'fuselage',
                               'node_in_upstream':0},
                     'winglet_l':{'upstream_component':'wing_l',
                               'node_in_upstream':10},
                     'vertical_tail':{'node2add':0,
                               'upstream_component':'fuselage',
                                      'node_in_upstream':10},
                     'horizontal_tail_right':{'node2add':0,
                               'upstream_component':'vertical_tail',
                                              'node_in_upstream':10},
                     'horizontal_tail_left':{'node2add':0,
                               'upstream_component':'vertical_tail',
                                      'node_in_upstream':10}
                     }
        }

g1sm = {'sharpy': {'simulation_input':None,
                   'default_module':'sharpy.routines.basic',
                   'default_solution':'sol_0',
                   'default_solution_vars': {'panels_wake':16*5,
                                             'AerogridPlot':{'include_rbm': 'off',
                                                             'include_applied_forces': 'off',
                                                             'minus_m_star': 0},
                                             'BeamPlot' : {'include_rbm': 'off',
                                                           'include_applied_forces': 'off'}},
                   'default_sharpy':{},
                   'model_route':None}}

g1 = gm.Model('sharpy',['sharpy'], model_dict=g1mm, components_dict=g1c,
               simulation_dict=g1sm)
#g1.build()
g1.run()
                                
