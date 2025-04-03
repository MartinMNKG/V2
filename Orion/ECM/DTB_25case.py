from Tools import *
from matplotlib.lines import Line2D
import pandas as pd 

Detailed_gas = ct.Solution("../detailed.yaml")
Reduced_gas = ct.Solution("../reduced.yaml")
Optim_A = ct.Solution("../STEC_A.yaml")
Optim_B = ct.Solution("../STEC_B.yaml")
pressure = np.linspace(1,1,1).tolist()
temperature = np.linspace(1000,2000,5).tolist()
phi = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
mixture =np.linspace(0.85,0.85,1).tolist()

Targets = ["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2','H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]
Non_Target =["AR"]

fuel1 = "NH3"
fuel2 ="H2"
oxidizer = 'O2:0.21, N2:0.79, AR : 0.01' 
tmax = 0.05
dt= 1e-6
case = generate_test_cases_bifuel(temperature,pressure,phi,mixture)

idx_target_det = spcs_name_idx(Detailed_gas,Targets)
idx_target_red = spcs_name_idx(Reduced_gas,Targets)
idx_non_target_det = spcs_name_idx(Detailed_gas,Non_Target)
idx_non_target_red = spcs_name_idx(Reduced_gas,Non_Target)

idx_target_OptimA = spcs_name_idx(Optim_A,Targets)
idx_target_OptimB = spcs_name_idx(Optim_B,Targets)
idx_non_target_OptimA = spcs_name_idx(Optim_A,Non_Target)
idx_non_target_OptimB = spcs_name_idx(Optim_B,Non_Target)

Time_det , Temp_det, Y_Target_det,Y_Non_Target_det = Sim0D_launch(Detailed_gas,Detailed_gas,fuel1,fuel2,oxidizer,case,idx_target_det,idx_non_target_det,dt,tmax)

with open(f"Time_det_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Time_det, file)
with open(f"Temp_det_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Temp_det, file)
with open(f"Y_Target_det_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Target_det, file)
with open(f"Y_Non_target_det_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Non_Target_det, file)


print("Launch Red")  
Time_red , Temp_red, Y_Target_red,Y_Non_Target_red = Sim0D_launch(Reduced_gas,Reduced_gas,fuel1,fuel2,oxidizer,case,idx_target_red,idx_non_target_red,dt,tmax) 


with open(f"Time_red_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Time_red, file)
with open(f"Temp_red_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Temp_red, file)
with open(f"Y_Target_red_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Target_red, file)
with open(f"Y_Non_target_red_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Non_Target_red, file)

print("Launch OptimA")  
Time_red , Temp_red, Y_Target_red,Y_Non_Target_red = Sim0D_launch(Optim_A,Optim_A,fuel1,fuel2,oxidizer,case,idx_target_red,idx_non_target_red,dt,tmax) 


with open(f"Time_OptimA_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Time_red, file)
with open(f"Temp_OptimA_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Temp_red, file)
with open(f"Y_Target_OptimA_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Target_red, file)
with open(f"Y_Non_target_OptimA_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Non_Target_red, file)
    

print("Launch OptimB")  
Time_red , Temp_red, Y_Target_red,Y_Non_Target_red = Sim0D_launch(Optim_B,Optim_B,fuel1,fuel2,oxidizer,case,idx_target_red,idx_non_target_red,dt,tmax) 


with open(f"Time_OptimB_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Time_red, file)
with open(f"Temp_OptimB_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Temp_red, file)
with open(f"Y_Target_OptimB_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Target_red, file)
with open(f"Y_Non_target_OptimB_cases_{len(case)}.pkl", "wb") as file:
    pickle.dump(Y_Non_Target_red, file)


