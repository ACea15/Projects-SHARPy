"""Script to read the wing deflections from Pickle files
Date: 24/05/22
Author: Pablo de Felipe
"""

import numpy as np
import os
import pdb
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra
importlib.reload(gm)
import pandas as pd
import pickle
from sharpy.utils.stochastic import Iterations
import sharpy.utils.generate_cases as gc
import sharpy.utils.h5utils as h5utils
import sharpy.utils.solver_interface as solver_interface
import sharpy.sharpy_main
import matplotlib
import matplotlib.pyplot as plt

importlib.reload(gm)
import sys

# Set the folder structure in case it is already not set
foldername = 'aoa_wingDef_investigation'
dirname = os.getcwd()
targetpath = os.path.join(dirname,foldername)

if os.path.exists(targetpath):
    print('Folder already exists!')# No need to do anything
else:
    os.mkdir(foldername)

try:
    model_route = os.path.dirname(os.path.realpath(__file__)) + '/'+foldername
except:

    import inspect

    __file__ = inspect.getfile(lambda: None)
    model_route = os.path.dirname(__file__) + '/'+foldername
def comp_settings(wing_length,
                  winglt_length,
                  wing_chord,
                  winglt_chord,
                  eiy,
                  m_bar_main,
                  ex,
                  components=['fuselage', 'wing_r', 'winglet_r',
                              'wing_l', 'winglet_l', 'vertical_tail',
                              'horizontal_tail_right', 'horizontal_tail_left'],
                  bound_panels=8):
    """ Function defining the parts of the model
    Args:
        wing_length   - Wing semispan without including the winglet in m
        wing_chord    - 1x2 nparray = [root chord, tip chord] in m
        winglt_chord  - 1x2 nparray = [root chord, tip chord] in m
        components    - List of components
        bound_panels  - bound panels on the wing and other lifting surfaces
    Returns:
        g1c_output    - Dictionary with all the data on components of the model
        """
    # aeroelasticity parameters
    main_ea = ex  # Wing elastic axis from LE as %
    main_cg = 0.3  # Not sure about this input
    sigma = 1.5

    #########
    # wings #
    #########
    #
    ea = 1e7
    ga = 1e5
    gj = 1e4
    #eiy = 2e4
    eiz = 4e6
    #m_bar_main = 0.75
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
                     'geometry': {'length': wing_length,
                                  'num_node': 13,
                                  'direction': [0., 1., 0.],
                                  'sweep': 0. * np.pi / 180,
                                  'dihedral': 0.},
                     'fem': {'stiffness_db': stiffness_main,
                             'mass_db': mass_main,
                             'frame_of_reference_delta': [-1, 0., 0.]},
                     'aero': {'chord': wing_chord,
                              'elastic_axis': main_ea,
                              'surface_m': bound_panels}
                     }
    g1c['winglet_r'] = {'workflow': ['create_structure', 'create_aero'],
                        'geometry': {'length': winglt_length,
                                     'num_node': 5,
                                     'direction': [0., 1., 0.],
                                     'sweep': 0. * np.pi / 180,
                                     'dihedral': 20. * np.pi / 180},
                        'fem': {'stiffness_db': stiffness_main,
                                'mass_db': mass_main,
                                'frame_of_reference_delta': [-1, 0., 0.]},
                        'aero': {'chord': winglt_chord,
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

def define_sol_0():
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
    return sol_0

def define_sol_112(u_inf,AoA_deg,rho,bound_panels):
    #############################################
    #  Aeroelastic equilibrium                  #
    #############################################
    AoA = AoA_deg * np.pi / 180
    c_ref = 1.0
    sol_112 = {
        'sharpy': {'simulation_input': None,
                   'default_module': 'sharpy.routines.static',
                   'default_solution': 'sol_112',
                   'default_solution_vars': {
                       'u_inf': u_inf,
                       'rho': rho,
                       'gravity_on': True,
                       'dt': c_ref / bound_panels / u_inf,
                       'panels_wake': bound_panels * 5,
                       'rotationA': [0., AoA, 0.],
                       'horseshoe': True,
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
    return sol_112
def define_sol_152(u_inf,AoA_deg,rho,bound_panels):
    c_ref = 1.0
    AoA = AoA_deg * np.pi / 180
    sol_152 = {'sharpy': {'simulation_input': None,
                          'default_module': 'sharpy.routines.flutter',
                          'default_solution': 'sol_152',
                          'default_solution_vars': {
                              'flutter_reference': 20.,
                              'root_method': 'bisection',
                              'velocity_increment': 10.,
                              'flutter_error': 0.001,
                              'damping_tolerance': 5e-3,
                              'inout_coordinates': 'modes',
                              'secant_max_calls': 15,
                              'rho': rho,
                              'gravity_on': True,
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
                              'add2_flow': [['StaticCoupled', 'plot','Pickle']],
                          },
                          'default_sharpy': {},
                          'model_route': None}}
    return sol_152

def linear_wingWeight(w1, taper, Sref):
    """Function to calculate the linear weight of the wing in kg/m

    Args:
        w1    - Constant for the wing
        taper - taper ratio
        Sref  - Wing Area in m^2

    Returns:
        linear_wingWeight
    """
    linear_wingWeight = w1 * taper ** 0.05 / (Sref) ** 0.5
    return linear_wingWeight

####################FUNCTION DEFINITIONS END###################################
# Define the constants
Sref = 32. #m^2
dhdrlSpanFraction = 0.25    # Ratio of each dihedral wing section to semispan
winglt_dhdrl  = 20*np.pi/180
w1 = 3*(2**0.5)                 # Constant for wing weight
#ex = 0.25                      # Elastic axis
eiy = 2e4
u_inf = 10.                     # Keep the speed constant!
rho = 1.225                     # Could be more precise at 1.225 kg/m^3

bound_panels = 4

# Calculate the wing semispan for different aspect ratios
#AR = np.array((10,20,30,40))
#AR = np.array((10,20))
#AR = np.array((20,25,35))
#AR  = np.array((30,))
#AR = np.linspace(10,40,7)
aoa_deg = np.array([0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7])
ar = np.array([32.,])
taper = np.array([1.,])
# Define variables to iterate
ex = np.array([0.25,0.3])

# Calculate variables which depend on AR
wing_span = np.sqrt(ar*Sref)
#wing_semispan = 0.5*AR*wing_chord-winglt_length*np.cos(winglt_dhdrl)
wing_semispan = wing_span*0.5
wing_length = wing_semispan*(1-dhdrlSpanFraction)
winglt_length = dhdrlSpanFraction*wing_semispan

# Calculate variables which depend on taper
chord_root = np.zeros([len(ar),len(taper)])
chord_wlt_root = np.zeros([len(ar),len(taper)])
chord_tip = np.zeros([len(ar),len(taper)])

for i in range(len(ar)):
    chord_root[i,:] = Sref/(wing_semispan[i]*(1+taper))
    chord_wlt_root[i,:] = chord_root[i,:]*(1+(taper-1)*wing_length[i]/wing_semispan[i])
    chord_tip[i,:] = chord_root[i,:]*taper

# Create the variables to iterate
# This is not the general way of implementing
iteration_dict = {
    'iterate_vars':{'aoa': aoa_deg,
                    'ar':ar,
                    'taper':taper,
                    'ex':ex
                    },
    'iterate_type': 'Full_Factorial',
    'iterate_labels':{'label_type':'number',
                      'print_name_var':0}
}

# Class that contains the details on the iteration
iteration = Iterations(iteration_dict['iterate_vars'],
                       iteration_dict['iterate_type'])
num_models = iteration.num_combinations
model_labels = iteration.labels(**iteration_dict['iterate_labels'])
dict2iterate = iteration.get_combinations_dict()
# Assign dependent variables
u_flutter = np.zeros((num_models,))

# Assign independent variables
aoa_pandas = np.zeros((num_models,))
ar_pandas = np.zeros((num_models,))
taper_pandas = np.zeros((num_models,))
ex_pandas = np.zeros((num_models,))
# eiy_pandas = np.zeros((num_models,))
for mi in range(num_models):
# Get the index number for each of the variables (using alphabetical order)
    i = int(model_labels[mi].split("_")[0])  # First variable being AoA
    j = int(model_labels[mi].split("_")[1])  # Second variable being AR
    k = int(model_labels[mi].split("_")[2])  # Third variable being Taper
    l = int(model_labels[mi].split("_")[3])  # Fourth variable being ex
    # n = int(model_labels[mi].split("_")[4])  # Fith variable being eiy (stiffness)
    aoa_pandas[mi] = aoa_deg[i]
    ar_pandas[mi] = ar[j]
    taper_pandas[mi] = taper[k]
    ex_pandas[mi] = ex[l]
    # eiy_pandas[mi] = eiy[n]

    # For this single built model create the files required
    folder2write = targetpath + '/' + foldername + '%s' % model_labels[mi]
    file2write = targetpath + '/' + foldername + '%s' % model_labels[mi] + '/' + foldername + '%s' % model_labels[mi]
    file2check = file2write + '/forces'
    errorpath = file2write+'/error.log'
    if os.path.exists(file2check):
        print('File exists!')
        # Open the pickle file and get the wing deflection without running the code
        os.chdir(file2write)
        infile = open(foldername + '%s' % model_labels[mi] + '.pkl', 'rb')
        data = pickle.load(infile)
        infile.close()
        # Obtain coordinates but initialise data storage
        if mi == 0:
            # Get the coordinates of the deformed config
            coordinates = data.structure.timestep_info[0].pos
            xc = coordinates[:, 0]
            yc = coordinates[:, 1]
            zc = coordinates[:, 2]
            # Obtain the wing coordinates
            xc_ini = data.structure.fortran['pos_ini'][:, 0]
            yc_ini = data.structure.fortran['pos_ini'][:, 1]
            # Find where the wing coordinates are
            condition = np.array([(xc_ini == 0) & (yc_ini >= 0)])[0, :]
            wing_coors = np.where(condition == True)[0]
            num_nodes = len(wing_coors)
            # Create arrays to store the wing y-z data
            y = np.zeros([num_nodes,num_models])
            z = np.zeros([num_nodes,num_models])
            # Store the data
            y[:,mi] = yc[wing_coors]
            z[:,mi] = zc[wing_coors]
            # Go back to the surrogate directory
            os.chdir('/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/flutter_investigation/')
        else:
            coordinates = data.structure.timestep_info[0].pos
            xc = coordinates[:, 0]
            yc = coordinates[:, 1]
            zc = coordinates[:, 2]
            y[:, mi] = yc[wing_coors]
            z[:, mi] = zc[wing_coors]

    else:
        g1 = gm.Model('sharpy', ['sharpy'],
                      model_dict=model_settings(foldername + '%s' % model_labels[mi]),
                      components_dict=comp_settings(wing_length[j],
                                                    winglt_length[j],
                                                    [chord_root[j, k], chord_wlt_root[j, k]],
                                                    [chord_wlt_root[j, k], chord_tip[j, k]],
                                                    eiy,
                                                    linear_wingWeight(w1, taper[k], Sref),
                                                    ex[l],
                                                    bound_panels=bound_panels),
                      simulation_dict=define_sol_112(u_inf, aoa_deg[i], rho, bound_panels))
        # simulation_dict=define_sol_0())
        # Create the file structure inside the folder
        g1.build()  # Build the model
        m = 0
        g1.built_models[m].sharpy.sim = gm.Simulation(sim_type='sharpy',
                                                      settings_sim=g1.simulation_dict['sharpy'],
                                                      case_route=folder2write,
                                                      case_name=g1.model_dict['model_name'])
        g1.built_models[m].sharpy.sim.get_sharpy(
            inp=g1.simulation_dict['sharpy']['simulation_input'])
        g1.built_models[m].sharpy.write_structure(file2write + '.fem.h5')
        g1.built_models[m].sharpy.write_aero(file2write + '.aero.h5')
        g1.built_models[m].sharpy.write_sim(file2write + '.sharpy')

        # data = g1build.run()
        # For some reason data is empty, have tried data[0] in notebook but does not help
        # As a fix run it manually
        print(os.getcwd())
        os.chdir('./' + foldername + '/' + foldername + '%s' % model_labels[mi])

        data = sharpy.sharpy_main.main(['', foldername + '%s.sharpy' % model_labels[mi]])

        # Store the values from the data file
        coordinates = data.structure.timestep_info[0].pos
        if mi == 0:
            # Get the coordinates of the deformed config
            xc = coordinates[:, 0]
            yc = coordinates[:, 1]
            zc = coordinates[:, 2]
            # Obtain the wing coordinates
            xc_ini = data.structure.fortran['pos_ini'][:, 0]
            yc_ini = data.structure.fortran['pos_ini'][:, 1]
            # Find where the wing coordinates are
            condition = np.array([(xc_ini == 0) & (yc_ini >= 0)])[0, :]
            wing_coors = np.where(condition == True)[0]
            num_nodes = len(wing_coors)
            # Create arrays to store the wing y-z data
            y = np.zeros([num_nodes, num_models])
            z = np.zeros([num_nodes, num_models])
            # Store the data
            y[:, mi] = yc[wing_coors]
            z[:, mi] = zc[wing_coors]
            # Go back to the surrogate directory
            os.chdir('/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/flutter_investigation/')
        else:
            xc = coordinates[:, 0]
            yc = coordinates[:, 1]
            zc = coordinates[:, 2]
            y[:, mi] = yc[wing_coors]
            z[:, mi] = zc[wing_coors]
            # Go back to the surrogate directory
            os.chdir('/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/flutter_investigation/')


print(y[:,4])
fig,ax = plt.subplots()
ax.plot(y[:,0],z[:,0],'b-x')
ax.plot(y[:,4],z[:,4],'r-o')
ax.plot(y[:,8],z[:,8],'g-^')


ax.legend(["AoA=0.0deg","AoA=1.2deg","AoA=2.4deg"])
ax.grid(True)
plt.axis('equal')
ax.set_ylabel('y (m)')
ax.set_xlabel('z (m)')
plt.show()

# Save data in a dictionary for comparison
data = {'aoa_pandas':aoa_pandas,
        'y':y,
        'z':z,
        }
# open a file, where you ant to store the data
save_path = targetpath+'/gravityOnData.pickle'
file = open(save_path, 'wb')
pickle.dump(data,file)
file.close()