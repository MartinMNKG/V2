from utils.Database import generate_test_cases,generate_test_cases_bifuel
import cantera as ct 
import os 
import glob
import numpy as np
from natsort import natsorted

# =====================================================================
#     LAUNCH SPEED FLAME // 0D REACTOR // COUNTERFLOW ==> Classification
# =====================================================================
def launch_speedflame(temperautre: list,pressure : list,eq_ratio :list,ref : str, classification :str,fuel :str,main_path : str):
    test_cases = generate_test_cases(temperautre,pressure,eq_ratio)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio = case
        flame = solve_speed_flame(gas_detailed,pressure,temperature,equivalence_ratio,nb_species_max,fuel,main_path,"Detailed")
    
    all_csv = glob.glob(os.path.join(classification,"*.yaml"))
    all_csv = natsorted(all_csv)
    for csv in reversed(all_csv) : 
        gas_reduced = ct.Solution(csv)
        nb_species= gas_reduced.n_species
        for case in test_cases : 
            pressure, temperature, equivalence_ratio = case
            flame = solve_speed_flame(gas_reduced,pressure,temperature,equivalence_ratio,nb_species,fuel,main_path,"Reduced")

        # MK ADD : LORS QUE TOUTE LES FLAMMES SONT FAITE POUR UN CSV : CALCUL D'ERR 
        # launch_error(nb_species_reduced)
            
def launch_0Dreactor(temperautre: list,pressure : list,eq_ratio :list,ref : str, classification :str,fuel :str,main_path,dt,tmax,mode):    
    test_cases = generate_test_cases(temperautre,pressure,eq_ratio)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio = case
        flame = run_0d_reactor(gas_detailed,pressure,temperature,equivalence_ratio,dt,tmax,nb_species_max,fuel,main_path,mode,"Detailed")
   
    
    all_csv = glob.glob(os.path.join(classification,"*.yaml"))
    all_csv = natsorted(all_csv)
    for csv in reversed(all_csv) : 
        gas_reduced = ct.Solution(csv)
        nb_species= gas_reduced.n_species
        for case in test_cases : 
            pressure, temperature, equivalence_ratio = case
            flame = run_0d_reactor(gas_reduced,pressure,temperature,equivalence_ratio,dt,tmax,nb_species,fuel,main_path,mode,"Reduced")
  
def launch_counterflow(temperautre: list,pressure : list,strain :list,ref : str, classification :str,fuel :str,main_path : str):
    test_cases = generate_test_cases(temperautre,pressure,strain)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, st = case
    
        flame = run_counterflow(gas_detailed,pressure,temperature,st,nb_species_max,fuel,main_path,"Detailed")
   
        # all_data_ref=launch_change_data(nb_species_max)
    
    all_csv = glob.glob(os.path.join(classification,"*.yaml"))
    all_csv = natsorted(all_csv)
    for csv in reversed(all_csv) : 
        gas_reduced = ct.Solution(csv)
        nb_species= gas_reduced.n_species
        for case in test_cases : 
            pressure, temperature, st = case
            flame = run_counterflow(gas_reduced,pressure,temperature,st,nb_species,fuel,main_path,"Reduced")
   
        # MK ADD : LORS QUE TOUTE LES FLAMMES SONT FAITE POUR UN CSV : CALCUL D'ERR 
        # launch_error(nb_species_reduced) 



# =====================================================================
#     LAUNCH SPEED FLAME // 0D REACTOR // COUNTERFLOW ==> Comparision/ Pure Fuel 
# =====================================================================
def launch_speedflame_pure(temperautre: list,pressure : list,eq_ratio :list,ref : str, red :str,fuel :str,main_path):
    print("Launch SpeedFlame databse")
    test_cases = generate_test_cases(temperautre,pressure,eq_ratio)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio = case
        flame = solve_speed_flame(gas_detailed,pressure,temperature,equivalence_ratio,nb_species_max,fuel,main_path,"Detailed")
    
    
    gas_reduced = ct.Solution(red)
    nb_species = gas_reduced.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio = case
        flame = solve_speed_flame(gas_reduced,pressure,temperature,equivalence_ratio,nb_species,fuel,main_path,"Reduced")
  
def launch_0Dreactor_pure(temperautre: list,pressure : list,eq_ratio :list,ref : str, red :str,fuel :str,main_path,dt,tmax,mode):
    print("Launch 0D Reactor database")
    test_cases = generate_test_cases(temperautre,pressure,eq_ratio)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio = case
        flame = run_0d_reactor(gas_detailed,pressure,temperature,equivalence_ratio,dt,tmax,nb_species_max,fuel,main_path,mode,"Detailed")
   
    
    gas_reduced = ct.Solution(red)
    nb_species = gas_reduced.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio = case
        flame = run_0d_reactor(gas_reduced,pressure,temperature,equivalence_ratio,dt,tmax,nb_species,fuel,main_path,mode,"Reduced")
   
def launch_counterflow_pure(temperautre: list,pressure : list,strain :list,ref : str, red :str,fuel :str,main_path):
    print("Launch Counterflow database")
    test_cases = generate_test_cases(temperautre,pressure,strain)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, st = case
        flame = run_counterflow(gas_detailed,pressure,temperature,st,nb_species_max,fuel,main_path,"Detailed")
   
    
    gas_reduced = ct.Solution(red)
    nb_species = gas_reduced.n_species
    for case in test_cases : 
        pressure, temperature, st = case
        flame = run_counterflow(gas_reduced,pressure,temperature,st,nb_species,fuel,main_path,"Reduced")
  
# =====================================================================
#     LAUNCH SPEED FLAME // 0D REACTOR // COUNTERFLOW ==> Comparision/ Bi-Fuel 
# =====================================================================
def launch_speedflame_bifuel(temperautre: list,pressure : list,eq_ratio :list, mixture : list ,ref : str, red :str,fuel :str, second_fuel : str ,main_path):
    print("Launch Bi fuel SpeedFlame databse")
    test_cases = generate_test_cases_bifuel(temperautre,pressure,eq_ratio,mixture)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio, mixture = case
        flame = solve_speed_flame_bifuel(gas_detailed,pressure,temperature,equivalence_ratio,mixture,nb_species_max,fuel,second_fuel,main_path,"Detailed")
    
    
    gas_reduced = ct.Solution(red)
    nb_species = gas_reduced.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio,mixture = case
        flame = solve_speed_flame_bifuel(gas_reduced,pressure,temperature,equivalence_ratio,mixture,nb_species,fuel,second_fuel,main_path,"Reduced")
  

def launch_0Dreactor_bifuel(temperature: list,pressure : list,eq_ratio :list,mixture: list,ref : str, red :str,fuel :str,second_fuel : str,main_path,dt,tmax,mode):
    print("Launch 0D Reactor database")
    test_cases = generate_test_cases_bifuel(temperature,pressure,eq_ratio,mixture)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, equivalence_ratio,mixture = case
        flame = run_0d_reactor_bifuel(gas_detailed,pressure,temperature,equivalence_ratio,mixture,dt,tmax,nb_species_max,fuel,second_fuel,main_path,mode,"Detailed")
    
    gas_reduced = ct.Solution(red)
    nb_species = gas_reduced.n_species
    print(nb_species)
    for case in test_cases : 
        pressure, temperature, equivalence_ratio,mixture = case
        flame = run_0d_reactor_bifuel(gas_reduced,pressure,temperature,equivalence_ratio,mixture,dt,tmax,nb_species,fuel,second_fuel,main_path,mode,"Reduced")

def launch_counterflow_bifuel(temperature : list, pressure : list, strain : list, mixture : list , ref : str , red : str , fuel : str, second_fuel :str, main_path : str) : 
    print("Launch Counterflow Database")
    test_cases = generate_test_cases_bifuel(temperature,pressure,strain,mixture)
    gas_detailed = ct.Solution(ref)
    nb_species_max = gas_detailed.n_species
    for case in test_cases : 
        pressure, temperature, strain,mixture = case 
        # flame = run_counterflow_bifuel(gas_detailed,pressure,temperature,mixture,nb_species_max,fuel,second_fuel,main_path,"Detailed")

    gas_reduced = ct.Solution(red)
    nb_species = gas_reduced.n_species 
    for case in test_cases : 
        pressure, temperature, strain,mixture = case 
        flame = run_counterflow_bifuel(gas_reduced,pressure,temperature,mixture,nb_species,fuel,second_fuel,main_path,"Reduced")
    
# =====================================================================
#      SPEED FLAME // 0D REACTOR // COUNTERFLOW ==> CANTERA / Pure Fuel
# =====================================================================
def solve_speed_flame(gas, pressure, temperature, equivalence_ratio, nb, fuel, main_path,type_f,transport_model="Mix"):
    # Créer un mélange de gaz initial
    
    
    fuel_ox_ratio = (
        gas.n_atoms(species=fuel, element="C")
        + 0.25 * gas.n_atoms(species=fuel, element="H")
        - 0.5 * gas.n_atoms(species=fuel, element="O")
    )

    compo = f"{fuel}:{equivalence_ratio:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}"

    gas.TPX = temperature, pressure, compo

    width = 0.06  # pour l'instant codée en dur a voir si modification

    # Créer la flamme
    flame = ct.FreeFlame(gas, width=width)
    flame.transport_model = transport_model

    # Initialisation de la solution
    flame.set_initial_guess()

    # Rafinement de la flamme
    flame.set_refine_criteria(ratio=2, slope=0.1, curve=0.2)

    # Résoudre la flamme
    flame.solve(loglevel=0, auto=True)

           
    dossier = os.path.join(main_path,f"{type_f}/{nb}S")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass

    flame.save(
        os.path.join(
            dossier, f"speedflame_{nb}S_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}.csv"
        ),
        basis="mole",
        overwrite=True,
    )

    return flame

def run_0d_reactor(gas,pressure,temperature,equivalence_ratio,dt,max_sim_time,nb,fuel,main_path,mode,type_f):
   
    equil_tol = 0.5

    # Créer un mélange de gaz initial
    gas.TP = temperature, pressure
    fuel_ox_ratio = (
        gas.n_atoms(species=fuel, element="C")
        + 0.25 * gas.n_atoms(species=fuel, element="H")
        - 0.5 * gas.n_atoms(species=fuel, element="O")
    )
    compo_ini = f"{fuel}:{equivalence_ratio:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}"
    

    gas.TPX = temperature, pressure, compo_ini

    # Créer le réacteur

    reactor = ct.ConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    

    # Donnée d'équilibre :
    gas_equil = gas
    gas_equil.TPX = temperature, pressure, compo_ini
    gas_equil.equilibrate("HP")
    state_equil = np.append(gas_equil.X, gas_equil.T)

    equil_bool = False
    n_iter = 0

    # Initialisation des résultats
    time = 0
    states = ct.SolutionArray(gas, extra=["t"])

    # Simulation
    print(mode)
    if mode == "equi":
        while equil_bool == False and time < max_sim_time:
            time += dt
            sim.advance(time)
            states.append(reactor.thermo.state, t=time)

            state_current = np.append(reactor.thermo.X, reactor.T)
            residual = (
                100
                * np.linalg.norm(state_equil - state_current, ord=np.inf)
                / np.linalg.norm(state_equil, ord=np.inf)
            )
            n_iter += 1

            # max iteration
            print(residual)
            if residual < equil_tol:
                equil_bool = True
                
    elif mode == "tmax":
        while time < max_sim_time:
            time += dt
            sim.advance(time)
            states.append(reactor.thermo.state, t=time)

            state_current = np.append(reactor.thermo.X, reactor.T)
            residual = (
                100
                * np.linalg.norm(state_equil - state_current, ord=np.inf)
                / np.linalg.norm(state_equil, ord=np.inf)
            )
            n_iter += 1

    # Sauvegarder les résultats
    dossier = os.path.join(main_path,f"{type_f}/{nb}S")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass

    states.write_csv(
        f"{dossier}/0Dreactor_{nb}S_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}.csv"
    )
    return time, states

def run_counterflow(gas, pressure, temperature,strain,nb,fuel,main_path,type_f):

    equivalence_ratio = 1
    fuel_ox_ratio = (
        gas.n_atoms(species=fuel, element="C")
        + 0.25 * gas.n_atoms(species=fuel, element="H")
        - 0.5 * gas.n_atoms(species=fuel, element="O")
    )
    compo_ini_o = f'O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
    compo_ini_f = f'{fuel}:{equivalence_ratio:3.2f}'

    gas.TPX = temperature, pressure, compo_ini_o
    density_o = gas.density
    
    gas.TPX = 300, pressure, compo_ini_f # Let temperature Fuel at 300 , only variate Oxy temperature ==> More realist
    density_f = gas.density

    loglevel = 0
    width = 0.02  # pour l'instant codée en dur a voir si modification

    f = ct.CounterflowDiffusionFlame(gas, width=width)
    f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)
    vel = strain * width/ 2.0 
    
    mdot_f = density_o*vel
    tin_f = temperature

    mdot_o = density_f*vel 
    tin_o = temperature

    # Set the state of the two inlets
    f.fuel_inlet.mdot = mdot_f
    f.fuel_inlet.X = compo_ini_f
    f.fuel_inlet.T = tin_f

    f.oxidizer_inlet.mdot = mdot_o
    f.oxidizer_inlet.X = compo_ini_o
    f.oxidizer_inlet.T = tin_o

    # Set the boundary emissivities
    f.boundary_emissivities = 0.0, 0.0
    # Turn radiation off
    f.radiation_enabled = False

    
    temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T) 
    # Solve the problem
    f.solve(loglevel, auto=True)

    # write the velocity, temperature, and mole fractions to a CSV file
    dossier = os.path.join(main_path,f"{type_f}/{nb}S")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass
    f.save(
        f"{dossier}/counterflow_{nb}S_ST{strain}_T{temperature}_P{pressure/101325}.csv",
        basis="mole",
        overwrite=True,
    )
    
    dossier_yaml = os.path.join(main_path,f"{type_f}/{nb}S/CF_{pressure}_{temperature}")
    if not os.path.exists(dossier_yaml) :
        os.makedirs(dossier_yaml)
    else : 
        print(f"dossier :{dossier_yaml} exist")
        pass 
    
    
    file_name, entry = names_yaml("initial-solution")
    f.save(os.path.join(dossier_yaml,file_name), name=entry, description="Initial solution",overwrite = True)




    # PART 2: COMPUTE EXTINCTION STRAIN

    # Exponents for the initial solution variation with changes in strain rate
    # Taken from Fiala and Sattelmayer (2014)
    exp_d_a = - 1. / 2.
    exp_u_a = 1. / 2.
    exp_V_a = 1.
    exp_lam_a = 2.
    exp_mdot_a = 1. / 2.

    # Set normalized initial strain rate
    alpha = [strain]
    # Initial relative strain rate increase
    delta_alpha = 1.
    # Factor of refinement of the strain rate increase
    delta_alpha_factor = 50.
    # Limit of the refinement: Minimum normalized strain rate increase
    delta_alpha_min = .1
    # Limit of the Temperature decrease
    delta_T_min = 1  # K

    # Iteration indicator
    n = 0
    # Indicator of the latest flame still burning
    n_last_burning = 0
    # List of peak temperatures
    T_max = [np.max(f.T)]
    # List of maximum axial velocity gradients
    a_max = [np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))]

    # Simulate counterflow flames at increasing strain rates until the flame is
    # extinguished. To achieve a fast simulation, an initial coarse strain rate
    # increase is set. This increase is reduced after an extinction event and
    # the simulation is again started based on the last burning solution.
    # The extinction point is considered to be reached if the abortion criteria
    # on strain rate increase and peak temperature decrease are fulfilled.
    while True:
        print(n)
        n += 1
        # Update relative strain rates
        alpha.append(alpha[n_last_burning] + delta_alpha)
        strain_factor = alpha[-1] / alpha[n_last_burning]
        # Create an initial guess based on the previous solution
        # Update grid
        # Note that grid scaling changes the diffusion flame width
        f.flame.grid *= strain_factor ** exp_d_a
        normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
        # Update mass fluxes
        f.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
        f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
        # Update velocities
        f.set_profile('velocity', normalized_grid,
                    f.velocity * strain_factor ** exp_u_a)
        f.set_profile('spread_rate', normalized_grid,
                    f.spread_rate * strain_factor ** exp_V_a)
        # Update pressure curvature
        f.set_profile('lambda', normalized_grid, f.L * strain_factor ** exp_lam_a)
        try:
            f.solve(loglevel=0)
        except ct.CanteraError as e:
            print('Error: Did not converge at n =', n, e)

        T_max.append(np.max(f.T))
        a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))
        if not np.isclose(np.max(f.T), temperature_limit_extinction):
            # Flame is still burning, so proceed to next strain rate
            n_last_burning = n
            file_name, entry = names_yaml(f"extinction/{n:04d}")
            f.save(os.path.join(dossier_yaml,file_name), name=entry, description=f"Solution at alpha = {alpha[-1]}",overwrite = True)
            
            file_name_csv, entry = names_csv(f"extinction/{n:04d}")
            f.save(os.path.join(dossier_yaml,file_name_csv), name=entry, description=f"Solution at alpha = {alpha[-1]}")
            
            print('Flame burning at alpha = {:8.4F}. Proceeding to the next iteration, '
                'with delta_alpha = {}'.format(alpha[-1], delta_alpha))
        elif ((T_max[-2] - T_max[-1] < delta_T_min) and (delta_alpha < delta_alpha_min)):
            # If the temperature difference is too small and the minimum relative
            # strain rate increase is reached, save the last, non-burning, solution
            # to the output file and break the loop
            
            file_name, entry = names_yaml(f"extinction/{n:04d}")
            f.save(os.path.join(dossier_yaml,file_name), name=entry, description=f"Flame extinguished at alpha={alpha[-1]}",overwrite = True)
            
            file_name_csv, entry = names_csv(f"extinction/{n:04d}")
            f.save(os.path.join(dossier_yaml,file_name_csv), name=entry, description=f"Flame extinguished at alpha={alpha[-1]}")

            print('Flame extinguished at alpha = {0:8.4F}.'.format(alpha[-1]),
                'Abortion criterion satisfied.')
            break
        else:
            # Procedure if flame extinguished but abortion criterion is not satisfied
            # Reduce relative strain rate increase
            delta_alpha = delta_alpha / delta_alpha_factor

            print('Flame extinguished at alpha = {0:8.4F}. Restoring alpha = {1:8.4F} and '
                'trying delta_alpha = {2}'.format(
                    alpha[-1], alpha[n_last_burning], delta_alpha))

            # Restore last burning solution
            file_name, entry = names_yaml(f"extinction/{n_last_burning:04d}")
            f.restore(os.path.join(dossier_yaml,file_name), entry)

    return f 



def names_yaml(test):
    # use separate files for YAML
    file_name = f"{test}.yaml".replace("-", "_").replace("/", "_")
    return file_name, "solution"

def names_csv(test):
    # use separate files for YAML
    file_name =f"{test}.csv".replace("-", "_").replace("/", "_")
    return file_name, "solution"
# =====================================================================
#      SPEED FLAME // 0D REACTOR // COUNTERFLOW ==> CANTERA / Bi-Fuel
# =====================================================================

def solve_speed_flame_bifuel(gas, pressure, temperature, equivalence_ratio,mixture, nb, fuel,second_fuel, main_path,type_f,transport_model="Mix"):
    # Créer un mélange de gaz initial
    
    fuel_composition = f'{fuel}:{mixture}, {second_fuel}:{1-mixture}'
    oxidizer_composition= f"O2 : 0.21 , N2 : 0.79" # Only O2 codée en dure pour l'instant
    
    #Dilution codée en dure pour l'instant 
    diluent_bool = False 
    if diluent_bool == True : 
        
        dilu = 'AR'
        frac = "diluent:0.7"
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition,basis="mole",diluent=dilu,fraction=frac)

    else : 
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition,basis="mole")

    gas.TP = temperature, pressure

    
    width = 0.06  # pour l'instant codée en dur a voir si modification

    # Créer la flamme
    flame = ct.FreeFlame(gas, width=width)
    flame.transport_model = transport_model

    # Initialisation de la solution
    flame.set_initial_guess()

    # Rafinement de la flamme
    flame.set_refine_criteria(ratio=2, slope=0.1, curve=0.1)

    # Résoudre la flamme
    flame.solve(loglevel=0, auto=True)

           
    dossier = os.path.join(main_path,f"{type_f}/{nb}S")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass

    flame.save(
        os.path.join(
            dossier, f"speedflame_{nb}S_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}_{fuel}_{mixture}.csv"
        ),
        basis="mole",
        overwrite=True,
    )

    return flame


def run_0d_reactor_bifuel(gas,pressure,temperature,equivalence_ratio,mixture,dt,max_sim_time,nb,fuel,second_fuel,main_path,mode,type_f):
   
    equil_tol = 0.5

    
    fuel_composition = f'{fuel}:{mixture}, {second_fuel}:{1-mixture}'
    oxidizer_composition= f"O2:0.21, N2 : 0.79" # Only O2 codée en dure pour l'instant
    
    #Dilution codée en dure pour l'instant 
    diluent_bool = False 
    if diluent_bool == True : 
        
        dilu = 'AR'
        frac = "diluent:0.7"
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition,basis="mole",diluent=dilu,fraction=frac)

    else : 
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition,basis="mole")

        
    gas.TP = temperature, pressure

    # Créer le réacteur

    reactor = ct.ConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    

    # Donnée d'équilibre :
    gas_equil = gas
    gas_equil.TP = temperature, pressure
    gas_equil.equilibrate("HP")
    state_equil = np.append(gas_equil.X, gas_equil.T)

    equil_bool = False
    n_iter = 0

    # Initialisation des résultats
    time = 0
    states = ct.SolutionArray(gas, extra=["t"])

    # Simulation
    print(mode)
    if mode == "equi":
        while equil_bool == False and time < max_sim_time:
            time += dt
            sim.advance(time)
            states.append(reactor.thermo.state, t=time)

            state_current = np.append(reactor.thermo.X, reactor.T)
            residual = (
                100
                * np.linalg.norm(state_equil - state_current, ord=np.inf)
                / np.linalg.norm(state_equil, ord=np.inf)
            )
            n_iter += 1
            print(residual)
            # max iteration
            # print(residual)
            if residual < equil_tol:
                equil_bool = True
                
    elif mode == "tmax":
        while time < max_sim_time:
            time += dt
            sim.advance(time)
            states.append(reactor.thermo.state, t=time)

            state_current = np.append(reactor.thermo.X, reactor.T)
            residual = (
                100
                * np.linalg.norm(state_equil - state_current, ord=np.inf)
                / np.linalg.norm(state_equil, ord=np.inf)
            )
            n_iter += 1

    # Sauvegarder les résultats
    dossier = os.path.join(main_path,f"{type_f}/{nb}S")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass

    states.write_csv(
        f"{dossier}/0Dreactor_{nb}S_ER{equivalence_ratio}_T{temperature}_P{pressure/101325}_{fuel}_{mixture}.csv"
    )
    return time, states

def names_yaml(test,path):
    # use separate files for YAML
    file_name = f"{path}/{test}.yaml"
    return file_name, "solution"

def names_csv(test,path):
        # use separate files for YAML
    file_name = f"{path}/{test}.csv"
    return file_name,"solution"


def run_counterflow_bifuel(gas,pressure,temperature,mixture,nb,fuel,second_fuel,main_path,type_f) : 
        
    
    width = 1.0
    f = ct.CounterflowDiffusionFlame(gas, width=width)
    tol_ss    = [1.0e-6, 1.0e-9]        # [rtol atol] for steady-state problem
    tol_ts    = [1.0e-6, 1.0e-9]        # [rtol atol] for time stepping
    
    fuel_composition = f'{fuel}:{mixture}, {second_fuel}:{1-mixture}'
    f.flame.set_steady_tolerances(default=tol_ss)
    f.flame.set_transient_tolerances(default=tol_ts)
    
    f.P = 1.e5  # 1 bar
    f.fuel_inlet.mdot = 0.305  # kg/m^2/s
    f.fuel_inlet.X = fuel_composition
    f.fuel_inlet.T = 300  # K
    f.oxidizer_inlet.mdot = 0.1  # kg/m^2/s
    f.oxidizer_inlet.X = 'O2:0.21, N2 : 0.78, AR: 0.01'
    f.oxidizer_inlet.T = temperature  # K
    f.energy_enabled = True

    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1, prune=0.03)
    temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T)
    f.solve(loglevel=0, auto=True)
    
    dossier = os.path.join(main_path,f"{type_f}/{nb}S/counterflow_P0_{pressure}_T0_{temperature}")
    if not os.path.exists(dossier): 
        os.makedirs(dossier)
    else :
        print(f"{dossier} already exist")
        pass
    
    file_name, entry = names_yaml("initial_solution",dossier)
    f.save(file_name, name=entry, description="Initial solution")

    file_name_csv, entry = names_csv("initial_solution",dossier)
    f.save(file_name_csv, name=entry, description="Initial solution")
    
    # PART 2: COMPUTE EXTINCTION STRAIN

    # Exponents for the initial solution variation with changes in strain rate
    # Taken from Fiala and Sattelmayer (2014)
    exp_d_a = - 1. / 2.
    exp_u_a = 1. / 2.
    exp_V_a = 1.
    exp_lam_a = 2.
    exp_mdot_a = 1. / 2.

    # Set normalized initial strain rate
    alpha = [100.]
    # Initial relative strain rate increase
    delta_alpha = 100.
    # Factor of refinement of the strain rate increase
    delta_alpha_factor = 10.
    # Limit of the refinement: Minimum normalized strain rate increase
    delta_alpha_min = 1
    # Limit of the Temperature decrease
    delta_T_min = 50  # K
    
    # Iteration indicator
    n = 0
    # Indicator of the latest flame still burning
    n_last_burning = 0
    # List of peak temperatures
    T_max = [np.max(f.T)]
    # List of maximum axial velocity gradients
    a_max = [np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))]

    while True : 
        n += 1
        # Update relative strain rates
        alpha.append(alpha[n_last_burning] + delta_alpha)
        strain_factor = alpha[-1] / alpha[n_last_burning]
        # Create an initial guess based on the previous solution
        # Update grid
        # Note that grid scaling changes the diffusion flame width
        f.flame.grid *= strain_factor ** exp_d_a
        normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
        # Update mass fluxes
        f.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
        f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
        # Update velocities
        f.set_profile('velocity', normalized_grid,
                    f.velocity * strain_factor ** exp_u_a)
        f.set_profile('spread_rate', normalized_grid,
                    f.spread_rate * strain_factor ** exp_V_a)
        # Update pressure curvature
        f.set_profile('lambda', normalized_grid, f.L * strain_factor ** exp_lam_a)
        try:
            f.solve(loglevel=0)
        except ct.CanteraError as e:
            print('Error: Did not converge at n =', n, e)
        
        
        T_max.append(np.max(f.T))
        a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))
        if not np.isclose(np.max(f.T), temperature_limit_extinction):
            # Flame is still burning, so proceed to next strain rate
            n_last_burning = n
            file_name, entry = names_yaml(f"extinction_{n:04d}",dossier)
            f.save(file_name, name=entry, description=f"Solution at alpha = {alpha[-1]}")

            file_name_csv, entry = names_csv(f"extinction_{n:04d}",dossier)
            f.save(file_name_csv, name=entry, description=f"Solution at alpha = {alpha[-1]}")
            
            print('Flame burning at alpha = {:8.4F}. Proceeding to the next iteration, '
                'with delta_alpha = {}'.format(alpha[-1], delta_alpha))
        
        elif (delta_alpha < delta_alpha_min):
        # If the temperature difference is too small and the minimum relative
        # strain rate increase is reached, save the last, non-burning, solution
        # to the output file and break the loop
            file_name, entry = names_yaml(f"extinction_{n:04d}",dossier)
            f.save(file_name, name=entry, description=f"Flame extinguished at alpha={alpha[-1]}")
            
            file_name_csv, entry = names_csv(f"extinction_{n:04d}",dossier)
            f.save(file_name_csv, name=entry, description=f"Flame extinguished at alpha={alpha[-1]}")

            print('Flame extinguished at alpha = {0:8.4F}.'.format(alpha[-1]),
                'Abortion criterion satisfied.')
            break
        else:
        # Procedure if flame extinguished but abortion criterion is not satisfied
        # Reduce relative strain rate increase
            delta_alpha = delta_alpha / delta_alpha_factor

            print('Flame extinguished at alpha = {0:8.4F}. Restoring alpha = {1:8.4F} and '
                'trying delta_alpha = {2}'.format(
                    alpha[-1], alpha[n_last_burning], delta_alpha))

            # Restore last burning solution
            file_name, entry = names_yaml(f"extinction_{n_last_burning:04d}",dossier)
            f.restore(file_name, entry)
            

    # Print some parameters at the extinction point, after restoring the last burning
    # solution
    file_name, entry = names_yaml(f"extinction_{n_last_burning:04d}",dossier)
    f.restore(file_name, entry)

    return f