from deap import base, creator, tools, algorithms
import random 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Tools import * 
import time

fitness = "PyOptMECH"
start_time = time.time()

Detailed_gas = ct.Solution("detailed.yaml")
Reduced_gas = ct.Solution("reduced.yaml")
pressure = np.linspace(1,1,1).tolist()
temperature = np.linspace(1500,2000,2).tolist()
phi = np.linspace(0.7,1.5,2).tolist()
mixture = [0.85]
Targets = ["H2", "NH3", "O2", "OH","NO","NH","NH2","NNH"]
Non_Target = ['N2', 'H', 'O', 'H2O', 'HO2', 'N', 'N2H3', 'N2H2', 'HNO', 'NO2', 'N2O']

param = ["shift","interpol","scaler"]

fuel1 = "NH3"
fuel2 ="H2"
oxidizer = 'O2:0.21, N2:0.79' 
tmax = 0.2
dt= 1e-6

# Maximum deviation from the start
variation_percent = 0.2
num_individu,init_value_factor = get_factor_dim(Reduced_gas)
bounds = [(val * (1 - variation_percent), val * (1 + variation_percent)) for val in init_value_factor]

# Get info from det and red 
idx_target_det = spcs_name_idx(Detailed_gas,Targets)
idx_target_red = spcs_name_idx(Reduced_gas,Targets)

idx_non_target_det = spcs_name_idx(Detailed_gas,Non_Target)
idx_non_target_red = spcs_name_idx(Reduced_gas,Non_Target)


# Calcul Data Origin from Detailed 
case = generate_test_cases_bifuel(temperature,pressure,phi,mixture)

print("Launch REF DATABASE")
# No tranform
Time_det , Temp_det, Y_Target_det,Y_Non_Target_det = Sim0D_launch(Detailed_gas,fuel1,fuel2,oxidizer,case,idx_target_det,idx_non_target_det,dt,tmax)
#transform (Shift, Interp, Scal)
Time_det , Temp_det, Y_Target_det,Y_Non_Target_det, Scaler_Target_det,Scaler_Non_Target_det,AI_delay_det= Change_detailed(Time_det,Temp_det,Y_Target_det,Y_Non_Target_det,case,Targets,Non_Target,param) # transform 

def evaluate(individual) : 
    new_gas = rxns_yaml_arr_list2(Reduced_gas,individual)
    
    # No transform 
    Time_red , Temp_red, Y_Target_red,Y_Non_Target_red = Sim0D_launch(new_gas,fuel1,fuel2,oxidizer,case,idx_target_red,idx_non_target_red,dt,tmax) 
    #transform (Shift, Interp, Scal)
    Time_red , Temp_red, Y_Target_red,Y_Non_Target_red,AI_delay_red =Change_Reduced(Time_red,Temp_red,Y_Target_red,Y_Non_Target_red,case,Targets,Non_Target,Time_det,Scaler_Target_det,Scaler_Non_Target_det,param)
    
    
   ## Return Fit = PyOptMech() 
    if fitness =="Dummy" : 
        return fit_dummy(case, AI_delay_det, AI_delay_red)
    
    if fitness =="PyOptMECH" : 
        return fit_pyoptmec_test(Y_Target_red,Y_Target_det,Time_det,Y_Non_Target_red,Y_Non_Target_det,Temp_red,Temp_det,AI_delay_det,AI_delay_red,case)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)


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
    return individual,



toolbox = base.Toolbox()
toolbox.register("individual", lambda: creator.Individual(create_individual(bounds)))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend,alpha=0.5)

toolbox.register("mutate", bounded_mutation, bounds=bounds, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selBest )


stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", min)
stats.register("avg", lambda fits: sum(fits) / len(fits))
logbook = tools.Logbook()

#Genetic Algorithm Parameters
population_size = 10
num_generations = 10
crossover_prob = 1
mutation_prob = 1

# Run the Genetic Algorithm
population = toolbox.population(n=population_size)
result_population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=crossover_prob,
    mutpb=mutation_prob,
    ngen=num_generations,
    stats=stats,
    halloffame=None,
    verbose=True,
)

# Extract the best solution
best_individual = tools.selBest(result_population, k=1)[0]
# print("Best pre-exponential factors:", best_individual)
print("Best fitness:", best_individual.fitness.values[0])

opt_gas = rxns_yaml_arr_list2(Reduced_gas,best_individual)
write_yaml(opt_gas ,"opt.yaml")

# Plot the statistics
generations = logbook.select("gen")
min_fitness = logbook.select("min")
avg_fitness = logbook.select("avg")

plt.figure()
plt.plot(generations, min_fitness, label="Min Fitness", marker='o')
plt.plot(generations, avg_fitness, label="Avg Fitness", marker='x', linestyle='--')
plt.title("Fitness Progression Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.savefig("Ouput_mono_proc.png")

print(f"Time : {time.time()-start_time}")