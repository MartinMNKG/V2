from Tools import *
from matplotlib.lines import Line2D
import pandas as pd 
verbose = True
Detailed_gas = ct.Solution("detailed.yaml")
Reduced_gas = ct.Solution("reduced.yaml")
Optim_1 = ct.Solution("STEC_A.yaml")
Optim_100 = ct.Solution("STEC_B.yaml")
pressure = np.linspace(1,1,1).tolist()
temperature = np.linspace(1000,2000,11).tolist()
phi = np.round(np.linspace(0.5, 2.0, 8), 1).tolist()
mixture =np.linspace(0.85,0.85,1).tolist()


Targets = ["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2']
Non_Target = ['H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]

param = ["shift","interpol","log","scaler"]

fuel1 = "NH3"
fuel2 ="H2"
oxidizer = 'O2:0.21, N2:0.79, AR : 0.01' 
tmax = 1
dt= 1e-6
case = generate_test_cases_bifuel(temperature,pressure,phi,mixture)
idx_target_det = spcs_name_idx(Detailed_gas,Targets)
idx_target_red = spcs_name_idx(Reduced_gas,Targets)
idx_target_optim1 = spcs_name_idx(Optim_1,Targets)
idx_target_optim100 = spcs_name_idx(Optim_100,Targets)

idx_non_target_det = spcs_name_idx(Detailed_gas,Non_Target)
idx_non_target_red = spcs_name_idx(Reduced_gas,Non_Target)
idx_non_target_optim1 = spcs_name_idx(Optim_1,Non_Target)
idx_non_target_optim100 = spcs_name_idx(Optim_100,Non_Target)

if verbose == True :
    print("Launch det")
    Time_det , Temp_det, Y_Target_det,Y_Non_Target_det = Sim0D_launch(Detailed_gas,Detailed_gas,fuel1,fuel2,oxidizer,case,idx_target_det,idx_non_target_det,dt,tmax)
    
    Time_det , Temp_det, Y_Target_det,Y_Non_Target_det, Scaler_Target_det,Scaler_Non_Target_det,AI_delay_det,Temp_scaler = Change_detailed(Time_det,Temp_det,Y_Target_det,Y_Non_Target_det,case,Targets,Non_Target,param) # transform 

    with open(f"Time_det_log_norm.pkl", "wb") as file:
        pickle.dump(Time_det, file)
    with open(f"Temp_det_log_norm.pkl", "wb") as file:
        pickle.dump(Temp_det, file)
    with open(f"Y_Target_det_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Target_det, file)
    with open(f"Y_Non_target_det_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Non_Target_det, file)
    with open(f"AI_det_log_norm.pkl", "wb") as file:
        pickle.dump(AI_delay_det, file)
    with open(f"Scaler_Y_Target_log_norm.pkl", "wb") as file:
        pickle.dump(Scaler_Target_det, file)
    with open(f"Scaler_Y_Non_Target_log_norm.pkl", "wb") as file:
        pickle.dump(Scaler_Non_Target_det, file)
    with open(f"Scaler_Temp_log_norm.pkl", "wb") as file:
        pickle.dump(Temp_scaler, file)
else : 
    Time_det = pd.read_pickle("Time_det.pkl")
    Temp_det = pd.read_pickle("Temp_det.pkl")
    Y_Target_det = pd.read_pickle("Y_Target_det.pkl")
    Y_Non_Target_det = pd.read_pickle("Y_Non_target_det.pkl")
    AI_delay_det = pd.read_pickle("AI_det.pkl")
    Scaler_Target_det = pd.read_pickle("Scaler_Y_Target.pkl")
    Scaler_Non_Target_det =pd.read_pickle("Scaler_Y_Non_Target.pkl")
    Temp_scaler=pd.read_pickle("Scaler_Temp.pkl")

if verbose == True : 
    print("Launch Red")  
    Time_red , Temp_red, Y_Target_red,Y_Non_Target_red = Sim0D_launch(Reduced_gas,Reduced_gas,fuel1,fuel2,oxidizer,case,idx_target_red,idx_non_target_red,dt,tmax) 
 
    Time_red , Temp_red, Y_Target_red,Y_Non_Target_red,AI_delay_red =Change_Reduced(Time_red,Temp_red,Y_Target_red,Y_Non_Target_red,case,Targets,Non_Target,Time_det,Scaler_Target_det,Scaler_Non_Target_det,Temp_scaler,param)


    with open(f"Time_red_log_norm.pkl", "wb") as file:
        pickle.dump(Time_red, file)
    with open(f"Temp_red_log_norm.pkl", "wb") as file:
        pickle.dump(Temp_red, file)
    with open(f"Y_Target_red_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Target_red, file)
    with open(f"Y_Non_target_red_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Non_Target_red, file)
    with open(f"AI_red_log_norm.pkl", "wb") as file:
        pickle.dump(AI_delay_red, file)
else : 
    Time_red = pd.read_pickle("Time_red.pkl")
    Temp_red = pd.read_pickle("Temp_red.pkl")
    Y_Target_red = pd.read_pickle("Y_Target_red.pkl")
    Y_Non_Target_red = pd.read_pickle("Y_Non_target_red.pkl")
    AI_delay_red = pd.read_pickle("AI_red.pkl")
        
if verbose == True : 
    print("Launch OPTIM 1")
    Time_red_optim1, Temp_red_optim1, Y_Target_red_optim1,Y_Non_Target_red_optim1 = Sim0D_launch(Optim_1,Optim_1,fuel1,fuel2,oxidizer,case,idx_target_optim1,idx_non_target_optim1,dt,tmax) 

    Time_red_optim1 , Temp_red_optim1, Y_Target_red_optim1,Y_Non_Target_red_optim1,AI_delay_red_optim1 =Change_Reduced(Time_red_optim1,Temp_red_optim1,Y_Target_red_optim1,Y_Non_Target_red_optim1,case,Targets,Non_Target,Time_det,Scaler_Target_det,Scaler_Non_Target_det,Temp_scaler,param)

    with open(f"Time_red_optim1_log_norm.pkl", "wb") as file:
        pickle.dump(Time_red_optim1, file)
    with open(f"Temp_red_optim1_log_norm.pkl", "wb") as file:
        pickle.dump(Temp_red_optim1, file)
    with open(f"Y_Target_red_optim1_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Target_red_optim1, file)
    with open(f"Y_Non_target_red_optim1_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Non_Target_red_optim1, file)
    with open(f"AI_red_optim1_log_norm.pkl", "wb") as file:
        pickle.dump(AI_delay_red_optim1, file)
        
else : 
    Time_red_optim1 = pd.read_pickle("Time_red_optim1.pkl")
    Temp_red_optim1 = pd.read_pickle("Temp_red_optim1.pkl")
    Y_Target_red_optim1 = pd.read_pickle("Y_Target_red_optim1.pkl")
    Y_Non_Target_red_optim1 = pd.read_pickle("Y_Non_target_red_optim1.pkl")
    AI_delay_red_optim1 = pd.read_pickle("AI_red_optim1.pkl")

if verbose == True :
    print("Launch Optim 2") 
    Time_red_optim100, Temp_red_optim100, Y_Target_red_optim100,Y_Non_Target_red_optim100 = Sim0D_launch(Optim_100,Optim_100,fuel1,fuel2,oxidizer,case,idx_target_optim100,idx_non_target_optim100,dt,tmax) 
    Time_red_optim100 , Temp_red_optim100, Y_Target_red_optim100,Y_Non_Target_red_optim100,AI_delay_red_optim100 =Change_Reduced(Time_red_optim100,Temp_red_optim100,Y_Target_red_optim100,Y_Non_Target_red_optim100,case,Targets,Non_Target,Time_det,Scaler_Target_det,Scaler_Non_Target_det,Temp_scaler,param)

    with open(f"Time_red_optim2_log_norm.pkl", "wb") as file:
        pickle.dump(Time_red_optim100, file)
    with open(f"Temp_red_optim2_log_norm.pkl", "wb") as file:
        pickle.dump(Temp_red_optim100, file)
    with open(f"Y_Target_red_optim2_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Target_red_optim100, file)
    with open(f"Y_Non_target_red_optim2_log_norm.pkl", "wb") as file:
        pickle.dump(Y_Non_Target_red_optim100, file)
    with open(f"AI_red_optim2_log_norm.pkl", "wb") as file:
        pickle.dump(AI_delay_red_optim100, file)
else : 
    Time_red_optim100 = pd.read_pickle("Time_red_optim2.pkl")
    Temp_red_optim100 = pd.read_pickle("Temp_red_optim2.pkl")
    Y_Target_red_optim100 = pd.read_pickle("Y_Target_red_optim2.pkl")
    Y_Non_Target_red_optim100 = pd.read_pickle("Y_Non_target_red_optim2.pkl")
    AI_delay_red_optim100 = pd.read_pickle("AI_red_optim2.pkl")
    
    

def fit_pyoptmec(Y_t_red,Y_t_det,time_det,Y_nt_red,Y_nt_det,temp_red,temp_det,ai_det,ai_red,case) : 
    F1_m =[]
    F2_m =[]
    F3_m =[] 
    F4_m = []
    for c in range(len(case)) : 
        # Equation F1_m
        top1 = np.trapezoid(np.abs(np.array(Y_t_red[c]) - np.array(Y_t_det[c])), np.array(time_det[c]))
        bot1 = np.trapezoid(np.abs(np.array(Y_t_det[c])), np.array(time_det[c]))
        F1_m.append( (top1/bot1)**2)

        # Equation F2_m
        top2 = [(np.max(Y_nt_red[c],axis=1)[j] - np.max(Y_nt_det[c],axis=1)[j]) for j in range(len(np.max(Y_nt_det[c],axis=1))) ]
        bot2 = [np.max(Y_nt_det[c],axis=1)[j] for j in range(len(np.max(Y_nt_det[c],axis=1))) ]
    
        F2_m.append((np.array(top2)/np.array(bot2))**2)

        # Equation F3_m
        top3 = np.trapezoid(np.abs(temp_red[c] - temp_det[c]), time_det[c])
        bot3 = np.trapezoid(np.abs(temp_det[c]), time_det[c])
        F3_m .append(( top3/ bot3)**2)

        # Equation F4_m
        top4 = (ai_red[c] - ai_det[c])
        bot4= ai_det[c]
        F4_m.append((top4/bot4)**2)
        
    weight = [1,1,1,1]
    # print(f"F1 = {F1_m}")
    # print(f"F2 = {F2_m}")
    
    _err = (
                weight[0] * np.sum(F1_m)
                + weight[1] * np.sum(F2_m)
                + weight[2] * F3_m
                + weight[3] * F4_m
            )
    # print(f"Err : {_err}")

    err = np.sqrt(np.sum(_err))
    print(np.sqrt(np.sum(_err)))
    return err,F1_m,F2_m,F3_m,F4_m

print("Launch Fitness")
Err,F1,F2,F3,F4 = fit_pyoptmec(Y_Target_red,Y_Target_det,Time_det,Y_Non_Target_red,Y_Non_Target_det,Temp_red,Temp_det,AI_delay_det,AI_delay_red,case)
Err_1 ,F1_1,F2_1,F3_1,F4_1 = fit_pyoptmec(Y_Target_red_optim1,Y_Target_det,Time_det,Y_Non_Target_red_optim1,Y_Non_Target_det,Temp_red_optim1,Temp_det,AI_delay_det,AI_delay_red_optim1,case)
Err_100 ,F1_100,F2_100,F3_100,F4_100 = fit_pyoptmec(Y_Target_red_optim100,Y_Target_det,Time_det,Y_Non_Target_red_optim100,Y_Non_Target_det,Temp_red_optim100,Temp_det,AI_delay_det,AI_delay_red_optim100,case)

#F1
plt.figure()
positions = range(1, len(Targets) + 1)
for spec in range(len(Targets)) : 
    plt.boxplot((np.array(F1)[:,spec]), positions=[spec + 1],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='yellow', color='yellow'),
                medianprops=dict(color='black'))
    plt.boxplot(np.array(F1_1)[:,spec], positions=[spec + 1 + 0.3],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor="red", color="red"),
                medianprops=dict(color='black'))
    plt.boxplot(np.array(F1_100)[:,spec], positions=[spec + 1 + 0.6],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor="blue", color="blue"),
                medianprops=dict(color='black'))

x = [x+0.3 for x in positions]
plt.xticks(x, Targets, rotation=90)  
legend_elements = [
    
    Line2D([0], [0], color='yellow', lw=4, label='Reduced'),
    Line2D([0], [0], color='red', lw=4, label='Optim A'),
    Line2D([0], [0], color='blue', lw=4, label='Optim B')
] 
plt.legend(handles=legend_elements)
plt.yscale('log')
plt.ylabel("Error")
plt.title("F1")
plt.savefig("F1_Giovani_log_norm.png")


#F2
plt.figure()
positions = range(1, len(Non_Target) + 1)

for spec in range(len(Non_Target)) : 
    plt.boxplot((np.array(F2)[:,spec]), positions=[spec + 1],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='yellow', color='yellow'),
                medianprops=dict(color='black'))
    plt.boxplot(np.array(F2_1)[:,spec], positions=[spec + 1 + 0.3],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor="red", color="red"),
                medianprops=dict(color='black'))
    plt.boxplot(np.array(F2_100)[:,spec], positions=[spec + 1 + 0.6],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor="blue", color="blue"),
                medianprops=dict(color='black'))
x = [x+0.3 for x in positions]
plt.xticks(x, Non_Target, rotation=90)  
legend_elements = [
    
    Line2D([0], [0], color='yellow', lw=4, label='Reduced'),
    Line2D([0], [0], color='red', lw=4, label='Optim A'),
    Line2D([0], [0], color='blue', lw=4, label='Optim B')
] 
plt.legend(handles=legend_elements)
plt.yscale('log')
plt.ylabel("Error")
plt.title("F2")
plt.savefig("F2_Giovani_log_norm.png")
    
    
plt.figure()
# F3
positions = range(1, len(["T"]) + 1)
plt.boxplot(F3, positions=[1-0.3],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='yellow', color='yellow'),
                medianprops=dict(color='black'))
plt.boxplot(F3_1, positions=[1],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='red', color='red'),
                medianprops=dict(color='black'))
plt.boxplot(F3_100, positions=[1+0.3],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='blue', color='blue'),
                medianprops=dict(color='black'))
plt.xticks(positions, ["T"], rotation=90)  
legend_elements = [
    
    Line2D([0], [0], color='yellow', lw=4, label='Reduced'),
    Line2D([0], [0], color='red', lw=4, label='Optim A'),
    Line2D([0], [0], color='blue', lw=4, label='Optim B')
] 
plt.legend(handles=legend_elements)
plt.yscale('log')
plt.ylabel("Error")
plt.title("F3")
plt.savefig("F3_Giovani_log_norm.png")

# F4
plt.figure()
positions = range(1, len(["AI_delay"]) + 1)
plt.boxplot(F4, positions=[1-0.3],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='yellow', color='yellow'),
                medianprops=dict(color='black'))
plt.boxplot(F4_1, positions=[1],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='red', color='red'),
                medianprops=dict(color='black'))
plt.boxplot(F4_100, positions=[1+0.3],showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='blue', color='blue'),
                medianprops=dict(color='black'))
plt.xticks(positions, ["AI_delay"], rotation=90)  
legend_elements = [
    
    Line2D([0], [0], color='yellow', lw=4, label='Reduced'),
    Line2D([0], [0], color='red', lw=4, label='Optim A'),
    Line2D([0], [0], color='blue', lw=4, label='Optim B')
] 
plt.legend(handles=legend_elements)
plt.yscale('log')
plt.ylabel("Error")
plt.title("F4")
plt.savefig("F4_Giovani_log_norm.png")

