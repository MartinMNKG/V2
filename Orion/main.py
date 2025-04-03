from deap import base, creator, tools, algorithms
import time 
import random 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Tools import * 
from mpi4py import MPI
import pickle
import csv
import re 

def optim_prob(pop_size,ngen,mutpb,cxpb,elitism_size,file_detailed,file_reduced,fitness,mode) :

    ########################
    # Init MPI
    ########################
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    ########################
    # DATABASE OF FLAME 
    ########################
    Detailed_gas = ct.Solution(file_detailed)
    Reduced_gas = ct.Solution(file_reduced)
    pressure = np.linspace(1,1,1).tolist()
    temperature = np.linspace(1000,2000,5).tolist()
    phi = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
    mixture =np.linspace(0.85,0.85,1).tolist()
    #PyOtp 
    if fitness == "PyOptMECH" :
        Targets = ["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2']
        Non_Target = ['H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]
    #OptiSmoke
    if fitness =="OptiSmoke" :
        Targets = ["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2','H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]
        Non_Target =["AR"]
        
    if fitness =="ORCh": 
        Targets = ["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2','H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]
        Non_Target =["AR"]
        eps = 1e-12
        coefficients = {
            "NO": 6.0,
            "NH": 3.5,
            "NH2": 3.5,
            "NNH": 5.0,
            "H2": 3.0,
            "NH3": 3.0,
            "O2": 3.0,
            "OH": 3.0,
            "O": 3.0,
            "H": 3.0

        }
        
    param = ["shift","interpol","noscaler"]

    fuel1 = "NH3"
    fuel2 ="H2"
    oxidizer = 'O2:0.21, N2:0.79, AR : 0.01' 
    tmax = 1
    dt= 1e-6

    case = generate_test_cases_bifuel(temperature,pressure,phi,mixture)
    
    if rank == 0 :
        
        dir = f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}"
        if os.path.exists(dir) == False : 
            os.makedirs(dir)
        if os.path.exists(dir+"/mech/") == False :
            os.makedirs(dir+"/mech/")
        if os.path.exists(dir+"/hist/") == False :
            os.makedirs(dir+"/hist/")

    ########################
    # INFO FOR CHANGE + ERR
    ########################


    variation_percent = 0.1
    num_individu,init_value_factor = get_factor_dim_ln(Reduced_gas)
    bounds = [(val * (1 - variation_percent), val * (1 + variation_percent)) for val in init_value_factor]

    idx_target_det = spcs_name_idx(Detailed_gas,Targets)
    idx_target_red = spcs_name_idx(Reduced_gas,Targets)

    idx_non_target_det = spcs_name_idx(Detailed_gas,Non_Target)
    idx_non_target_red = spcs_name_idx(Reduced_gas,Non_Target)


        

    ########################
    # Init Creator and Tool Box
    ########################

    # Initialisation des individus avec des bornes spécifiques
    def create_gene(lower, upper):
        return random.uniform(lower, upper)

    def create_individual(bounds):
        return [create_gene(lower, upper) for lower, upper in bounds]

    # Check si la mutation reste dans les bornes 
    def bounded_mutation(individual, bounds, mu, sigma, indpb):
        # Apply Gaussian mutation
        tools.mutGaussian(individual, mu, sigma, indpb)
        # Enforce bounds
        for i, (lower, upper) in enumerate(bounds):
            individual[i] = max(min(individual[i], upper), lower)
        return individual


    def repair(individual):
        """Repairs individual values to ensure they stay within bounds."""
        for i, (lower, upper) in enumerate(bounds):
            individual[i] = max(min(individual[i], upper), lower)
        return individual

    # Def Evaluate ==> Here to change Fitness 
    def evaluate(individual) : 
        new_gas = rxns_yaml_arr_list2_ln(Reduced_gas,individual)
        new_gas_eq = rxns_yaml_arr_list2_ln(Reduced_gas,individual)
        # Time_det, Y_Target_det, Y_Non_Target_det, Temp_det, AI_delay_det, Scaler_Target_det, Scaler_Non_Target_det = load_ref(pop_size,ngen,mutpb)
        
        # No transform 
        Time_red , Temp_red, Y_Target_red,Y_Non_Target_red = Sim0D_launch(new_gas,new_gas_eq,fuel1,fuel2,oxidizer,case,idx_target_red,idx_non_target_red,dt,tmax) 
        #transform (Shift, Interp, Scal)
        Time_red , Temp_red, Y_Target_red,Y_Non_Target_red,AI_delay_red =Change_Reduced(Time_red,Temp_red,Y_Target_red,Y_Non_Target_red,case,Targets,Non_Target,Time_det,Scaler_Target_det,Scaler_Non_Target_det,Temp_scaler,param)
        
        
        ## Return Fit = PyOptMech() 
        if fitness =="Dummy" : 
            return fit_dummy(case, AI_delay_det, AI_delay_red)
        
        if fitness =="PyOptMECH" : 
            # return fit_pyoptmec(case,Y_Target_red,Y_Target_det,Time_det,Y_Non_Target_red,Y_Non_Target_det,Temp_red,Temp_red,AI_delay_red,AI_delay_det,weight=[1,1,1,1])
            # return fit_pyoptmec_test(Y_t_red,Y_t_det,time_det,Y_nt_red,Y_nt_det,temp_red,temp_det,ai_det,ai_red,case)
            return fit_pyoptmec_test(Y_Target_red,Y_Target_det,Time_det,Y_Non_Target_red,Y_Non_Target_det,Temp_red,Temp_det,AI_delay_det,AI_delay_red,case)

        if fitness =="OptiSmoke" :
            return fit_OptimSmoke(Y_Target_det,Y_Target_red,case)
        if fitness =="ORCh" : 
            return fit_orch(Y_Target_red,Y_Target_det,coefficients,Targets,eps,coef_non_target=0.05)
        
    def mpi_evaluate(population):
        # Diviser la population entre les processus
        chunk_size = len(population) // size
        if rank == size - 1:
            chunk = population[rank * chunk_size:]  # Dernier processus prend le reste
        else:
            chunk = population[rank * chunk_size:(rank + 1) * chunk_size]
        
        # Évaluer le sous-ensemble assigné à ce processus
        local_results = list(map(toolbox.evaluate, chunk))
        
        # Rassembler les résultats dans le processus maître
        gathered_results = comm.gather(local_results, root=0)
        
        # Appliquer les résultats aux individus
        if rank == 0:
            flat_results = [item for sublist in gathered_results for item in sublist]
            for ind, fit in zip(population, flat_results):
                ind.fitness.values = fit
        comm.barrier()  # Synchronisation
    

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual(create_individual(bounds)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend,alpha=0.5)
    toolbox.register("mutate", bounded_mutation, bounds=bounds, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)


    ## Here to add other info to stats 
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg",lambda fits: sum(fits) / len(fits))
    
    toolbox.register("map", mpi_evaluate) ## ne sais pas si c'est utile 

    def main() : 
        
        
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals","min","avg"]


        # Initialize and broadcast population
        
        if mode == "Start" : 
            ind_start = 1
            # population = toolbox.population(n=pop_size)
            population = toolbox.population(n=pop_size -1)
            special_inidivual = creator.Individual(init_value_factor)
            population.append(special_inidivual)
            mpi_evaluate(population)
            
            comm.barrier()
            
            if rank == 0 : 
                record = stats.compile(population)
                logbook.record(gen=0, nevals=len(population), **record)
            
            
        if mode =="Restart":
            hist_path = f"Start_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/hist/"
            files = [f for f in os.listdir(hist_path) if re.match(r'population_\d+\.pkl', f)]
            files.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Tri numérique
            last_population_file = files[-1]  # Prendre le dernier fichier

            # Charger la dernière population
            with open(os.path.join(hist_path, last_population_file), "rb") as f:
                population = pickle.load(f)
                ind_start = int(re.search(r'\d+', last_population_file).group())
                
                
                
        for gen in range(ind_start, ngen + 1):
        
            # Sélection
            if rank == 0:
                
                elite = tools.selBest(population, elitism_size)
            
                offspring = toolbox.select(population, len(population) - elitism_size )
                offspring = list(map(toolbox.clone, offspring))
            
            else:
                offspring = None
            
            # Broadcast des enfants à tous les processus
            offspring = comm.bcast(offspring, root=0)

            # Appliquer croisement et mutation
            
          
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
		    
             
                 
            for mutant in offspring:

                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                repair(mutant)

            # Réévaluer les individus modifiés
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            
            mpi_evaluate(invalid_ind)

            comm.barrier()
            # Rassembler les statistiques + maj pop
            if rank == 0:
                population[:] = invalid_ind + elite
                with open(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/hist/population_{gen}.pkl", "wb") as f:
                    pickle.dump(population, f)
                record = stats.compile(population)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                print(logbook.stream)
                save_best = tools.selBest(population, 1)[0]
                opt_gas = rxns_yaml_arr_list2_ln(Reduced_gas,save_best)
                write_yaml(opt_gas ,f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/mech/Mech_gen_{gen}.yaml")
                with open(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/hist/Output_mpi_gen{gen}.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(logbook.header)  # Write the header (columns)
                    for record in logbook:
                        writer.writerow(record.values())  # Write each generation's data
                
                
        return population, logbook


    if rank == 0 : 
        start_time = time.time()


        
        Time_det , Temp_det, Y_Target_det,Y_Non_Target_det = Sim0D_launch(Detailed_gas,Detailed_gas,fuel1,fuel2,oxidizer,case,idx_target_det,idx_non_target_det,dt,tmax)
        
        Time_det , Temp_det, Y_Target_det,Y_Non_Target_det, Scaler_Target_det,Scaler_Non_Target_det,AI_delay_det,Temp_scaler = Change_detailed(Time_det,Temp_det,Y_Target_det,Y_Non_Target_det,case,Targets,Non_Target,param) # transform 
        
        
        F_ref = evaluate(init_value_factor)
        opt_gas = rxns_yaml_arr_list2_ln(Reduced_gas,init_value_factor)
        write_yaml(opt_gas ,f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/Reduced_gen1_{fitness}.yaml")

        
        
        
        with open(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/F_ref.pkl", "wb") as file:
            pickle.dump(F_ref, file)
        

        

    else : 
        Time_det = None
        Temp_det = None
        Y_Target_det = None
        Y_Non_Target_det = None
        Scaler_Target_det = None
        Scaler_Non_Target_det = None
        AI_delay_det = None
        Temp_scaler = None
        
        F_ref = None 
        
    # comm.barrier()

    Time_det = comm.bcast(Time_det, root=0)
    Temp_det = comm.bcast(Temp_det, root=0)
    Y_Target_det = comm.bcast(Y_Target_det, root=0)
    Y_Non_Target_det = comm.bcast(Y_Non_Target_det, root=0)
    Scaler_Target_det = comm.bcast(Scaler_Target_det, root=0)
    Scaler_Non_Target_det = comm.bcast(Scaler_Non_Target_det, root=0)
    AI_delay_det = comm.bcast(AI_delay_det, root=0)
    Temp_scaler= comm.bcast(Temp_scaler,root=0)
    F_ref = comm.bcast(F_ref,root= 0)
    
   
        
    comm.barrier()

    result_population,logbook =main()

    if rank == 0 : 
    
        end_time = time.time()-start_time
        print(f"Time Optim:{end_time}")
        
        # Extract the best solution
        with open(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/time_log.txt", "w") as file:
            file.write(str(end_time))
        best_individual = tools.selBest(result_population, k=1)[0]
            

        # Assume 'logbook' is your DEAP logbook object
        
        with open(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/Loogbox_Optim_Pop{pop_size}_Gen{ngen}_{fitness}_Case_{len(case)}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(logbook.header)  # Write the header (columns)
            for record in logbook:
                writer.writerow(record.values())  # Write each generation's data

        
        print("Best fitness:", best_individual.fitness.values[0])
        # print(logbook)
        opt_gas = rxns_yaml_arr_list2_ln(Reduced_gas,best_individual)
        write_yaml(opt_gas ,f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/Reduced_mpi_pop_{fitness}.yaml")

        # Plot the statistics
        generations = logbook.select("gen")
        min_fitness = logbook.select("min")
        avg_fitness = logbook.select("avg")
        plt.figure()
        plt.plot(generations, min_fitness, label="Min Fitness", marker='o')
        # plt.plot(generations, avg_fitness, label="Avg Fitness", marker='x', linestyle='--')
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/Ouput_mpi_pop{pop_size}_gen{ngen}_{fitness}_cases_{len(case)}.png")
        plt.figure() 
        
        plt.figure()
        plt.plot(generations, min_fitness/F_ref[0], label="Min Fitness", marker='o')
        # plt.plot(generations, avg_fitness, label="Avg Fitness", marker='x', linestyle='--')
        plt.title("Fitness Progression Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{mode}_mpi_pop{pop_size}_gen{ngen}_{fitness}_Case{len(case)}/Ouput_mpi_pop{pop_size}_gen{ngen}_{fitness}_cases_{len(case)}_F_FREF.png")
        plt.figure() 
    

