import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Tools import *
from matplotlib.patches import Patch
        
def Generate_commun_grid(time,case,lenght):
    output = [] 
    for c in range(len(case)) : 
        loc_time = time[c]
        output.append(np.linspace(np.min(loc_time),np.max(loc_time),lenght)) 

    return output        
        
def Interpol_temp(time,temp,grid,case) : 
    output = [] 
    for c in range(len(case)) : 
        loc_time = time[c]
        loc_grid = grid[c]
        loc_temp = temp[c]
        int_func = interp1d(loc_time,loc_temp,fill_value="extrapolate")
        output.append(int_func(loc_grid))
    return output

def Interpol_Y(time,data,Commun_time,case,spec) : 
    output = []
    for c in range(len(case)):
        loc_time = time[c]
        loc_data = data[c]
        loc_commun_time = Commun_time[c]
        loc_output = []
        for s in range(len(spec)): 
    
            int_func = interp1d(loc_time,loc_data[:,s],fill_value="extrapolate")
            loc_output.append(int_func(loc_commun_time))
        output.append(loc_output)
    return output

def Standard_Y (data_ref,case,spec):
    output = [] 
    scaler = []
    for c in range(len(case)) : 
        loc_data_ref = data_ref[c]
        loc_output =[]
        loc_scaler = []
        
        for s in range(len(spec)) :  
            scl = MinMaxScaler() 
            scl.fit(loc_data_ref[s])
            loc_output.append(scl.transform(loc_data_ref[s]))
            loc_scaler.append(scl)
        output.append(loc_output)
        scaler.append(loc_scaler)
    return output,scaler 

def Standard_Y_rdrc(data_red,scaler,case,spec) : 
    output =[] 
    for c in range(len(case)) : 
        loc_output =[]
        loc_data_red = data_red[c]
        loc_scaler = scaler[c]
        for s in range(len(spec)) : 
            scl=loc_scaler[s] 
            loc_output.append(scl.transform(loc_data_red[s]))
        
        output.append(loc_output) 
    return output 

def Standard_T(data_ref,case): 
    output = [] 
    scaler = [] 
    for c in range(len(case)) : 
        scl = MinMaxScaler() 
        scl.fit(data_ref[c])
        output.append(scl.transform(data_ref[c]))
        scaler.append(scl)
    return output,scaler

def Standar_T_rdrc(data_red,scaler,case): 
    output = [] 
    for c in range(len(case)) : 
        scl = scaler[c]
        output.append(scl.transform(data_red[c]))
    return output 
  
def compute_ORCH(data_Y_det, data_Y_red,spec : list,coef,coef_non_target=0.05,eps=1e-12) : 
    Err = np.abs(np.array(data_Y_det)-np.array(data_Y_red))/np.maximum(np.abs(np.array(data_Y_det)),eps)
    mask = np.abs(data_Y_det)< eps
    Err[mask] = 0 
    output = [] 
    for s in spec : 
        ind = spec.index(s)
        if s in coef : 
            k=coef[s]
        else : 
            k = coef_non_target
        
        output.append(k*np.sum(np.array(Err[:, ind, :]).flatten()))
    return Err,output

def compute_PMO(cases, Interp_Y_det, Interp_Y_red, Interp_Temp_det, Interp_Temp_red, 
                   IDT_det, IDT_red, Commun_time, Targets, Intergrate_Species, Peak_species):
    """
    Compute different error metrics for species integration, peak values, temperature, and ignition delay time (IDT).
    
    Parameters:
        cases (list): List of cases.
        Interp_Y_det (list): Detailed species mass fractions interpolated.
        Interp_Y_red (list): Reduced species mass fractions interpolated.
        Interp_Temp_det (list): Detailed temperature profiles.
        Interp_Temp_red (list): Reduced temperature profiles.
        IDT_det (list): Detailed ignition delay times.
        IDT_red (list): Reduced ignition delay times.
        Commun_time (list): Common time grids for each case.
        Targets (list): List of species names.
        Intergrate_Species (list): List of species for integration-based error.
        Peak_species (list): List of species for peak-value-based error.
    
    Returns:
        tuple: (F1, F2, F3, F4) lists containing error metrics.
    """
    
    F1, F2, F3, F4 = [], [], [], []

    for c in range(len(cases)):  
        loc_time = Commun_time[c]
        loc_Y_det = Interp_Y_det[c]
        loc_Y_red = Interp_Y_red[c]
        loc_T_det = Interp_Temp_det[c]
        loc_T_red = Interp_Temp_red[c]
        loc_IDT_det = IDT_det[c]
        loc_IDT_red = IDT_red[c]

        loc_F1, loc_F2 = [], []

        for s in range(len(Targets)):
            loc_loc_Y_det = loc_Y_det[s]
            loc_loc_Y_red = loc_Y_red[s]

            if Targets[s] in Intergrate_Species:
                top1 = np.trapezoid(np.abs(np.array(loc_loc_Y_red) - np.array(loc_loc_Y_det)), np.array(loc_time))
                bot1 = np.trapezoid(np.abs(np.array(loc_loc_Y_red)), np.array(loc_time))
                loc_F1.append((top1 / bot1) ** 2 if bot1 != 0 else 0)

            elif Targets[s] in Peak_species:
                top2 = np.max(loc_loc_Y_det) - np.max(loc_loc_Y_red)
                bot2 = np.max(loc_loc_Y_det)
                loc_F2.append((top2 / bot2) ** 2 if bot2 != 0 else 0)

        F1.append(loc_F1)
        F2.append(loc_F2)

        top3 = np.trapezoid(np.abs(loc_T_red - loc_T_det), loc_time)
        bot3 = np.trapezoid(np.abs(loc_T_det), loc_time)
        F3.append((top3 / bot3) ** 2 if bot3 != 0 else 0)

        top4 = loc_IDT_red - loc_IDT_det
        bot4 = loc_IDT_det
        F4.append((top4 / bot4) ** 2 if bot4 != 0 else 0)

    return F1, F2, F3, F4

Targets = ["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2','H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]
Non_Target =["AR"]
pressure = np.linspace(1,1,1).tolist()
temperature = np.linspace(1000,2000,5).tolist()
phi = np.round(np.linspace(0.8, 1.2, 5), 1).tolist()
mixture =np.linspace(0.85,0.85,1).tolist()
case = generate_test_cases_bifuel(temperature,pressure,phi,mixture)

Temp_det =pd.read_pickle("Temp_det_cases_25.pkl")
Time_det =pd.read_pickle("Time_det_cases_25.pkl")
Y_Target_det = pd.read_pickle("Y_Target_det_cases_25.pkl")
Y_NonTarget_det =pd.read_pickle("Y_Non_target_det_cases_25.pkl")
Temp_red =pd.read_pickle("Temp_red_cases_25.pkl")
Time_red =pd.read_pickle("Time_red_cases_25.pkl")
Y_Target_red = pd.read_pickle("Y_Target_red_cases_25.pkl")
Y_NonTarget_red =pd.read_pickle("Y_Non_target_red_cases_25.pkl")
Temp_OptimA =pd.read_pickle("Temp_OptimA_cases_25.pkl")
Time_OptimA =pd.read_pickle("Time_OptimA_cases_25.pkl")
Y_Target_OptimA = pd.read_pickle("Y_Target_OptimA_cases_25.pkl")
Y_NonTarget_OptimA =pd.read_pickle("Y_Non_target_OptimA_cases_25.pkl")
Temp_OptimB =pd.read_pickle("Temp_OptimB_cases_25.pkl")
Time_OptimB =pd.read_pickle("Time_OptimB_cases_25.pkl")
Y_Target_OptimB = pd.read_pickle("Y_Target_OptimB_cases_25.pkl")
Y_NonTarget_OptimB =pd.read_pickle("Y_Non_target_OptimB_cases_25.pkl")


fitness= False
Time_shift= False  
log = False 
scaler= False

print(f"CALCULATE FITNESS = {fitness}")
print(f"SHIFT = {Time_shift}")
print(f"LOG = {log}")
print(f"SCALER = {scaler}")
#Calcul IDT 
 
IDT_det = Calc_ai_delay(Time_det,Temp_det,case)
IDT_red = Calc_ai_delay(Time_red,Temp_red,case)
IDT_OptimA = Calc_ai_delay(Time_OptimA,Temp_OptimA,case)
IDT_OptimB = Calc_ai_delay(Time_OptimB,Temp_OptimB,case)

#SHIFT :
if Time_shift == True : 
        
    for i in range(len(case)) : 
        Time_det[i] = np.array(Time_det[i]) - IDT_det[i]
        Time_red[i] = np.array(Time_red[i]) - IDT_red[i]
        Time_OptimA[i] = np.array(Time_OptimA[i]) - IDT_OptimA[i]
        Time_OptimB[i] = np.array(Time_OptimB[i]) - IDT_OptimB[i]
        
        
        
#LOG

if log == True : 
    Y_Target_det = np.log(Y_Target_det)
    Y_Target_red = np.log(Y_Target_red)
    Y_Target_OptimA = np.log(Y_Target_OptimA)
    Y_Target_OptimB = np.log(Y_Target_OptimB)
        
        
#Generate commun_grid     
Commun_time = Generate_commun_grid(Time_det,case,500)       
        
    
#Interpolation :
#Y    
Interp_Y_det = Interpol_Y(Time_det,Y_Target_det,Commun_time,case,Targets)
Interp_Y_red = Interpol_Y(Time_red,Y_Target_red,Commun_time,case,Targets)
Interp_Y_OptimA = Interpol_Y(Time_OptimA,Y_Target_OptimA,Commun_time,case,Targets)
Interp_Y_OptimB = Interpol_Y(Time_OptimB,Y_Target_OptimB,Commun_time,case,Targets)
        
        
#Temp
Interp_Temp_det = Interpol_temp(Time_det,Temp_det,Commun_time,case)
Interp_Temp_red = Interpol_temp(Time_red,Temp_red,Commun_time,case)
Interp_Temp_OptimA= Interpol_temp(Time_OptimA,Temp_OptimA,Commun_time,case)
Interp_Temp_OptimB = Interpol_temp(Time_OptimB,Temp_OptimB,Commun_time,case) 

        
        
#Scaler  
if scaler == True : 
    Interp_Y_det,Scal_Y_det = Standard_Y(Interp_Y_det,case,Targets)
    Interp_Y_red = Standard_Y_rdrc(Interp_Y_red,Scal_Y_det,case,Targets)
    Interp_Y_OptimA = Standard_Y_rdrc(Interp_Y_OptimA,Scal_Y_det,case,Targets)
    Interp_Y_OptimB= Standard_Y_rdrc(Interp_Y_OptimB,Scal_Y_det,case,Targets)

    Interp_Temp_det,Scal_T_det = Standard_T(Interp_Temp_det,case)
    Interp_Temp_red=Standar_T_rdrc(Interp_Temp_red,Scal_T_det,case)
    Interp_Temp_OptimA=Standar_T_rdrc(Interp_Temp_OptimA,Scal_T_det,case)
    Interp_Temp_OptimB=Standar_T_rdrc(Interp_Temp_OptimB,Scal_T_det,case)



## Err ABS : 

Err_Y_abs = np.abs(np.array(Interp_Y_det)-np.array(Interp_Y_red))
Err_T_abs = np.abs(np.array(Interp_Temp_det)-np.array(Interp_Temp_red))
Err_IDT_abs = np.abs(np.array(IDT_det)-np.array(IDT_red))

Err_Y_OptimA_abs = np.abs(np.array(Interp_Y_det)-np.array(Interp_Y_OptimA))
Err_T_OptimA_abs = np.abs(np.array(Interp_Temp_det)-np.array(Interp_Temp_OptimA))
Err_IDT_OptimA_abs = np.abs(np.array(IDT_det)-np.array(IDT_OptimA))

Err_Y_OptimB_abs = np.abs(np.array(Interp_Y_det)-np.array(Interp_Y_OptimB))
Err_T_OptimB_abs = np.abs(np.array(Interp_Temp_det)-np.array(Interp_Temp_OptimB))
Err_IDT_OptimB_abs = np.abs(np.array(IDT_det)-np.array(IDT_OptimB))

print(np.shape(Err_Y_abs))
print(np.shape(Err_Y_OptimA_abs[:,0,:]))

##########################################
#AE
##########################################

remove = ["N2","H","O","HO2","N","N2H2","HNO","NH","NH2","NNH","N2O"]#["H","O","HO2","N","N2H2","HNO","NH","NH2","NNH"]#["NH3","NO2","N2O","H","HO2","N","N2H2","HNO","NH","NH2","NNH"] #["NH3","N2H2"] ##Pour Shift Interpol et Min Max ### Pour Interpol Only 
# Décalages pour chaque type d'erreur (évite superposition)
offsets = [-0.3, 0., 0.3]  # Décalages pour les 3 courbes par groupe
colors = ['green', 'red', 'blue']
labels = ['Reduced', 'Optim A', 'Optim B']
# Boucle sur les variables de Target

New_Target=[]


plt.clf()
plt.figure()
plt.rcParams.update({'font.size': 18}) 
a = 0
for s in range(len(Targets)):
  
    data_list = [Err_Y_abs[:, s, :], Err_Y_OptimA_abs[:, s, :], Err_Y_OptimB_abs[:, s, :]]
    if Targets[s] not in remove :
        New_Target.append(Targets[s]) 
        for i, data in enumerate(data_list):
            plt.boxplot(
                np.array(data).flatten(),
                positions=[a + 1 + offsets[i]],
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=colors[i], color=colors[i]),
                medianprops=dict(color='black'),
                widths=0.2  # Réduit la largeur pour éviter chevauchement
            )
        a = a+1 


# Ajout d'une légende
legend_patches = [plt.Line2D([0], [0], color=col, marker='s', markersize=10, linestyle='None', label=lab) 
                  for col, lab in zip(colors, labels)]
plt.legend(handles=legend_patches, bbox_to_anchor=(1, 1.2),ncol=3, fontsize=14)

NT = [rf"$Y_{{{x}}}$" for x in New_Target]
plt.xticks(ticks=range(1, len(NT) + 1), labels=NT , rotation=45)
plt.ylabel(r"$\mathcal{E}_{mech}(\hat{Y}_i)$ distribution")  # Ajoute un label
# plt.title(r"Absolute Error ($\mathcal{E}_{mech}(Y_i)$) distribution for each scheme")  # Ajoute un titre
plt.yscale("log")
plt.savefig(f"AE_{len(case)}_cases_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png",bbox_inches="tight")
# Ajout de T

plt.figure()
plt.rcParams.update({'font.size': 18}) 
data_list_T = [Err_T_abs[:, :], Err_T_OptimA_abs[:, :], Err_T_OptimB_abs[:, :]]
for i, data in enumerate(data_list_T):
    plt.boxplot(
        np.array(data).flatten(),
        positions=[1 + offsets[i]],  # Décalé après Targets
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colors[i], color=colors[i]),
        medianprops=dict(color='black'),
        widths=0.2
    )
data_list_IDT = [Err_IDT_abs[:], Err_IDT_OptimA_abs[:], Err_IDT_OptimB_abs[:]]
for i, data in enumerate(data_list_IDT):
    plt.boxplot(
        np.array(data).flatten(),
        positions=[2 + offsets[i]],  # Décalé après T
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colors[i], color=colors[i]),
        medianprops=dict(color='black'),
        widths=0.2
    )
plt.xticks(ticks=[1,2], labels=[r"$T$","IDT"], rotation=0)
plt.ylabel(r"$\mathcal{E}_{mech}(T)$ and $\mathcal{E}_{mech}(IDT)$ distribution")  # Ajoute un label
# plt.title(r"Absolute Error ($\mathcal{E}_{mech}(T)$) distribution for each scheme")  # Ajoute un titre
plt.yscale("log")
# plt.ylim([1e-27,1e1])
legend_patches = [plt.Line2D([0], [0], color=col, marker='s', markersize=10, linestyle='None', label=lab) 
                  for col, lab in zip(colors, labels)]
plt.legend(handles=legend_patches, bbox_to_anchor=(1, 1.2),ncol=3, fontsize=14)
plt.savefig(f"AE_T_{len(case)}_cases_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png",bbox_inches="tight")



plt.figure()
plt.rcParams.update({'font.size': 18}) 
# Ajout de IDT
data_list_IDT = [Err_IDT_abs[:], Err_IDT_OptimA_abs[:], Err_IDT_OptimB_abs[:]]
for i, data in enumerate(data_list_IDT):
    plt.boxplot(
        np.array(data).flatten(),
        positions=[1 + offsets[i]],  # Décalé après T
        whis = (0,100),
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colors[i], color=colors[i]),
        medianprops=dict(color='black'),
        widths=0.2
    )
plt.ylabel(r"$\mathcal{E}_{mech}(IDT)$ distribution")  # Ajoute un label
# plt.title(r"Absolute Error ($\mathcal{E}_{mech}(IDT)$) distribution for each scheme")  # Ajoute un titre
plt.yscale("log")
plt.xticks(ticks=[1], labels=["IDT"], rotation=45)

legend_patches = [plt.Line2D([0], [0], color=col, marker='s', markersize=10, linestyle='None', label=lab) 
                  for col, lab in zip(colors, labels)]
plt.legend(handles=legend_patches, bbox_to_anchor=(1, 1.2),ncol=3, fontsize=14)
plt.savefig(f"AE_IDT_{len(case)}_cases_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png",bbox_inches="tight")




################

fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 14}) 
select_spec = "NH3"
ind_max_Orch = Targets.index(select_spec)
print(ind_max_Orch)
# Tracer les courbes de l'axe principal
ax1.plot(Commun_time[0], Interp_Y_det[0][ind_max_Orch], "k", label="Detailed")
ax1.plot(Commun_time[0], Interp_Y_red[0][ind_max_Orch], "g", label="Reduced")
ax1.plot(Commun_time[0], Interp_Y_OptimA[0][ind_max_Orch], "r", label="Optim A")
ax1.plot(Commun_time[0], Interp_Y_OptimB[0][ind_max_Orch], "b", label="Optim B")
ax1.set_xlabel("Time [s]",fontsize=16)
ax1.set_ylabel(r"$Y_{NH_3}$")

# Deuxième axe Y
ax2 = ax1.twinx()
ax2.plot(Commun_time[0], Err_Y_abs[0,ind_max_Orch,:], linestyle="--", color="darkgreen", label=r"$\mathcal{E}_{abs}^{Reduced}$") 
ax2.plot(Commun_time[0], Err_Y_OptimA_abs[0,ind_max_Orch,:], linestyle="--", color="darkred", label=r"$\mathcal{E}_{abs}^{Optim A}$") 
ax2.plot(Commun_time[0], Err_Y_OptimB_abs[0,ind_max_Orch,:], linestyle="--", color="darkblue", label=r"$\mathcal{E}_{abs}^{Optim B}$")  
ax2.set_ylabel(r"$\mathcal{E}_{abs}$")  
ax2.set_yscale("log")

ax2.spines["right"].set_linestyle('--')


# Récupérer les handles et labels des deux axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Créer un patch vide pour espacer
empty_patch = Patch(color='none', label="")

# Organisation en tableau (2 lignes, 4 colonnes)
handles = [handles1[0], handles1[1], handles1[2], handles1[3],  # Ligne 1
        empty_patch, handles2[0], handles2[1], handles2[2]]  # Ligne 2
labels = ["Detailed", "Reduced", "Optim A", "Optim B",  # Ligne 1
        "", r"$\mathcal{E}_{abs}^{Reduced}$", r"$\mathcal{E}_{abs}^{Optim A}$", r"$\mathcal{E}_{abs}^{Optim B}$"]  # Ligne 2

handles_2 = [handles1[0],empty_patch, handles1[1],handles2[0],handles1[2],handles2[1],handles1[3],handles2[2]]
labels_2 = ["Detailed","","Reduced", r"$\mathcal{E}_{abs}^{Reduced}$","Optim A",r"$\mathcal{E}_{ORCh}^{abs A}$","Optim B", r"$\mathcal{E}_{abs}^{Optim B}$"]

# Placer la légende sous forme de tableau
legend = fig.legend(handles_2, labels_2, loc="upper center", bbox_to_anchor=(0.5, 1.0), 
                    ncol=4, frameon=True)

# Ajuster l'espace en haut pour éviter que la légende soit coupée
plt.subplots_adjust(top=0.8)

# Sauvegarde
plt.savefig(f"Case_NH3_Error_Max_ABS_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", bbox_inches="tight")




##########################################
##########################################
if fitness == True :
    #ORCH

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

    Err_Orch,Err_Orch_coef = compute_ORCH(Interp_Y_det,Interp_Y_red,Targets,coefficients)
    Err_Orch_OptimA,Err_Orch_OptimA_coef = compute_ORCH(Interp_Y_det,Interp_Y_OptimA,Targets,coefficients)
    Err_Orch_OptimB,Err_Orch_OptimB_coef =compute_ORCH(Interp_Y_det,Interp_Y_OptimB,Targets,coefficients)
    print("Orch :")
    print(f"Reduced = {sum(Err_Orch_coef):.3e}")
    print(f"Optim A ={sum(Err_Orch_OptimA_coef):.3e}")
    print(f"Optim B = {sum(Err_Orch_OptimB_coef):.3e}")

    #PLOT FITNESS 
    plt.figure()
    plt.rcParams.update({'font.size': 18}) 
    bar_width = 0.3  
    x = np.arange(len(Targets))  
    plt.figure(figsize=(16, 6)) 
    plt.rcParams.update({'font.size': 18}) 
    plt.bar(x - bar_width, Err_Orch_coef, width=bar_width, label="Reduced", color='green')
    plt.bar(x, Err_Orch_OptimA_coef, width=bar_width, label="Optim A", color='red')
    plt.bar(x + bar_width, Err_Orch_OptimB_coef, width=bar_width, label="Optim B", color='blue')
    plt.xticks(ticks=x, labels=Targets, rotation=45)
    plt.yscale("log")
    plt.legend()
    plt.ylabel(r"$F_{obj}(Y^{i})$")
    plt.tight_layout()
    plt.ylim([1e5,1e9])
    plt.savefig(f"ORCH_{len(case)}_cases_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png")

    ## Plot avec filtre 
    # Filtrer les espèces dont toutes les valeurs sont en dessous du seuil 1e5
    seuil = 1e5
    filtered_indices = [
        i for i in range(len(Targets)) 
        if max(Err_Orch_coef[i], Err_Orch_OptimA_coef[i], Err_Orch_OptimB_coef[i]) >= seuil
    ]

    # Mise à jour des listes après filtrage
    Targets_filtered = [Targets[i] for i in filtered_indices]
    Err_Orch_coef_filtered = [Err_Orch_coef[i] for i in filtered_indices]
    Err_Orch_OptimA_coef_filtered = [Err_Orch_OptimA_coef[i] for i in filtered_indices]
    Err_Orch_OptimB_coef_filtered = [Err_Orch_OptimB_coef[i] for i in filtered_indices]

    # Mise à jour des indices pour le plot
    x = np.arange(len(Targets_filtered))

    # PLOT FITNESS 
    plt.figure()
    plt.rcParams.update({'font.size': 18}) 
    bar_width = 0.3  
    plt.figure(figsize=(16, 6)) 
    plt.rcParams.update({'font.size': 18}) 
    plt.bar(x - bar_width, Err_Orch_coef_filtered, width=bar_width, label="Reduced", color='green')
    plt.bar(x, Err_Orch_OptimA_coef_filtered, width=bar_width, label="Optim A", color='red')
    plt.bar(x + bar_width, Err_Orch_OptimB_coef_filtered, width=bar_width, label="Optim B", color='blue')
    
    NT_orch= [rf"$Y_{{{x}}}$" for x in Targets_filtered]
    plt.xticks(ticks=x, labels=NT_orch, fontsize="24",rotation=45)
    plt.yscale("log")
    plt.legend()
    plt.ylabel(r"$F_{obj}(Y^{i})$",fontsize="24")
    plt.tight_layout()
    plt.ylim([1e5,5e9])
    plt.savefig(f"ORCH_Filtered_{len(case)}_cases_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png")



    #Plot where error Max 
    select_spec ="NH3"
    ind_max_Orch = Targets.index(select_spec)
    Err_red=[]
    Err_A=[]
    Err_B = []
    for c in range(len(case)) : 
        Err_red.append(np.sum(Err_Orch[c][ind_max_Orch])*coefficients[select_spec])
        Err_A.append(np.sum(Err_Orch_OptimA[c][ind_max_Orch])*coefficients[select_spec])
        Err_B.append(np.sum(Err_Orch_OptimB[c][ind_max_Orch])*coefficients[select_spec])
        
    plt.figure()
    plt.rcParams.update({'font.size': 18}) 
    plt.plot(Err_red,"g",label="Reduced")
    plt.plot(Err_A,"r",label="Optim A")
    plt.plot(Err_B,'b',label="Optim B")
    plt.legend()
    # plt.legend(["Reduced", "Optim A", "Optim B"], loc='upper center', ncol=3, fontsize=16)
    plt.yscale("log")
    plt.ylabel(r"$F_{ORCh}(Y_{NH3})$",fontsize=16)
    plt.xlabel("cases",fontsize=16)

    plt.tight_layout()
    plt.savefig(f"NH3_Error_Max_ORCH_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png")


    ###########################################
    #Plot sim error Max 
    fig, ax1 = plt.subplots()
    plt.rcParams.update({'font.size': 14}) 
    # Tracer les courbes de l'axe principal
    ax1.plot(Commun_time[0], Interp_Y_det[0][ind_max_Orch], "k", label="Detailed")
    ax1.plot(Commun_time[0], Interp_Y_red[0][ind_max_Orch], "g", label="Reduced")
    ax1.plot(Commun_time[0], Interp_Y_OptimA[0][ind_max_Orch], "r", label="Optim A")
    ax1.plot(Commun_time[0], Interp_Y_OptimB[0][ind_max_Orch], "b", label="Optim B")
    ax1.set_xlabel("Time [s]",fontsize=16)
    ax1.set_ylabel(r"$Y_{NH_3}$")

    # Deuxième axe Y
    ax2 = ax1.twinx()
    ax2.plot(Commun_time[0], Err_Orch[0][ind_max_Orch]*coefficients[select_spec], linestyle="--", color="darkgreen", label=r"$\mathcal{E}_{ORCh}^{Reduced}$") 
    ax2.plot(Commun_time[0], Err_Orch_OptimA[0][ind_max_Orch]*coefficients[select_spec], linestyle="--", color="darkred", label=r"$\mathcal{E}_{ORCh}^{Optim A}$") 
    ax2.plot(Commun_time[0], Err_Orch_OptimB[0][ind_max_Orch]*coefficients[select_spec], linestyle="--", color="darkblue", label=r"$\mathcal{E}_{ORCh}^{Optim B}$")  
    ax2.set_ylabel(r"$\mathcal{E}_{ORCh}$")  
    ax2.set_yscale("log")

    ax2.spines["right"].set_linestyle('--')


    # Récupérer les handles et labels des deux axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Créer un patch vide pour espacer
    empty_patch = Patch(color='none', label="")

    # Organisation en tableau (2 lignes, 4 colonnes)
    handles = [handles1[0], handles1[1], handles1[2], handles1[3],  # Ligne 1
            empty_patch, handles2[0], handles2[1], handles2[2]]  # Ligne 2
    labels = ["Detailed", "Reduced", "Optim A", "Optim B",  # Ligne 1
            "", r"$\mathcal{E}_{ORCh}^{Reduced}$", r"$\mathcal{E}_{ORCh}^{Optim A}$", r"$\mathcal{E}_{ORCh}^{Optim B}$"]  # Ligne 2

    handles_2 = [handles1[0],empty_patch, handles1[1],handles2[0],handles1[2],handles2[1],handles1[3],handles2[2]]
    labels_2 = ["Detailed","","Reduced", r"$\mathcal{E}_{ORCh}^{Reduced}$","Optim A",r"$\mathcal{E}_{ORCh}^{Optim A}$","Optim B", r"$\mathcal{E}_{ORCh}^{Optim B}$"]

    # Placer la légende sous forme de tableau
    legend = fig.legend(handles_2, labels_2, loc="upper center", bbox_to_anchor=(0.5, 1.0), 
                        ncol=4, frameon=True)

    # Ajuster l'espace en haut pour éviter que la légende soit coupée
    plt.subplots_adjust(top=0.8)
    plt.rcParams.update({'font.size': 18}) 

    # Sauvegarde
    plt.savefig(f"Case_NH3_Error_Max_ORCH_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", bbox_inches="tight")


    ##########################################
    #PMO 
    ##########################################
    
    Intergrate_Species =["H2", "NH3", "O2", "OH","NO", 'H2O','NO2', 'N2O','N2']
    Peak_species = ['H', 'O', 'HO2', 'N', 'N2H2', 'HNO',"NH","NH2","NNH"]


    F1_red, F2_red, F3_red, F4_red = compute_PMO(case, Interp_Y_det, Interp_Y_red, Interp_Temp_det, Interp_Temp_red, 
                                    IDT_det, IDT_red, Commun_time, Targets, Intergrate_Species, Peak_species)


    F1_OptimA, F2_OptimA, F3_OptimA, F4_OptimA = compute_PMO(case, Interp_Y_det, Interp_Y_OptimA, Interp_Temp_det, Interp_Temp_OptimA, 
                                    IDT_det, IDT_OptimA, Commun_time, Targets, Intergrate_Species, Peak_species)


    F1_OptimB, F2_OptimB, F3_OptimB, F4_OptimB = compute_PMO(case, Interp_Y_det, Interp_Y_OptimB, Interp_Temp_det, Interp_Temp_OptimB, 
                                    IDT_det, IDT_OptimB, Commun_time, Targets, Intergrate_Species, Peak_species)

    print("PMO")
    print(f"Reduced = {np.sqrt(np.sum(F1_red)+np.sum(F2_red)+np.sum(F3_red)+np.sum(F4_red)):.3e}")
    print(f"Optim A ={np.sqrt(np.sum(F1_OptimA)+np.sum(F2_OptimA)+np.sum(F3_OptimA)+np.sum(F4_OptimA)):.3e}")
    print(f"Optim B = {np.sqrt(np.sum(F1_OptimB)+np.sum(F2_OptimB)+np.sum(F3_OptimB)+np.sum(F4_OptimB)):.3e}")
    
    print(f"Reduced : F1 = {np.sum(F1_red)} ,F2 = {np.sum(F2_red)}, F3 = {np.sum(F3_red)}, F4 = {np.sum(F4_red)}")
    print(f"Optim A : F1 = {np.sum(F1_OptimA)} ,F2 = {np.sum(F2_OptimA)}, F3 = {np.sum(F3_OptimA)}, F4 = {np.sum(F4_OptimA)}")
    print(f"Reduced : F1 = {np.sum(F1_OptimB)} ,F2 = {np.sum(F2_OptimB)}, F3 = {np.sum(F3_OptimB)}, F4 = {np.sum(F4_OptimB)}")
    # Définir une taille plus petite pour s'ajuster à une page A4
    fig_width = 6  # largeur en pouces
    fig_height = 3  # hauteur en pouces


    # Graphique 1
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    xF1 = np.arange(len(Intergrate_Species))
    ax1.bar(xF1 - bar_width, np.sum(F1_red, axis=0), width=bar_width, label="Reduced", color='green')
    ax1.bar(xF1, np.sum(F1_OptimA, axis=0), width=bar_width, label="Optim A", color='red')
    ax1.bar(xF1 + bar_width, np.sum(F1_OptimB, axis=0), width=bar_width, label="Optim B", color='blue')
    
    NT_F1= [rf"$Y_{{{x}}}$" for x in Intergrate_Species]
    ax1.set_xticks(xF1)
    ax1.set_xticklabels(NT_F1, rotation=45)
    ax1.set_yscale("log")
    ax1.set_ylim([1e0, 1e3])
    ax1.set_ylabel(r"$F_1(Y_i)$")
    ax1.legend(loc='upper center', ncol=3, fontsize=10)
    plt.rcParams.update({'font.size': 18}) 
    # Enregistrer la première figure
    fig1.savefig(f"F1_plot_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", dpi=300, bbox_inches='tight')
    
    
    ###############
    
    # Définition du seuil
    seuil = 1e0

    # Filtrer les indices où au moins une valeur dépasse le seuil
    filtered_indices = [
        i for i in range(len(Intergrate_Species)) 
        if max(np.sum(F1_red, axis=0)[i], np.sum(F1_OptimA, axis=0)[i], np.sum(F1_OptimB, axis=0)[i]) >= seuil
    ]

    # Mise à jour des listes filtrées
    Intergrate_Species_filtered = [Intergrate_Species[i] for i in filtered_indices]
    F1_red_filtered = np.sum(F1_red, axis=0)[filtered_indices]
    F1_OptimA_filtered = np.sum(F1_OptimA, axis=0)[filtered_indices]
    F1_OptimB_filtered = np.sum(F1_OptimB, axis=0)[filtered_indices]

    # Mise à jour des indices pour le plot
    xF1 = np.arange(len(Intergrate_Species_filtered))

    # Création du graphe filtré
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax1.bar(xF1 - bar_width, F1_red_filtered, width=bar_width, label="Reduced", color='green')
    ax1.bar(xF1, F1_OptimA_filtered, width=bar_width, label="Optim A", color='red')
    ax1.bar(xF1 + bar_width, F1_OptimB_filtered, width=bar_width, label="Optim B", color='blue')

    # Configuration des axes et légendes
    ax1.set_xticks(xF1)
    ax1.set_xticklabels(Intergrate_Species_filtered, rotation=45)
    ax1.set_yscale("log")
    ax1.set_ylim([1e0, 1e3])
    ax1.set_ylabel(r"$F_1(Y_i)$")
    ax1.legend(loc='upper center', ncol=3, fontsize=10)
    plt.rcParams.update({'font.size': 18}) 
    # Enregistrement de la figure
    fig1.savefig(f"F1_Filtered_plot_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", dpi=300, bbox_inches='tight')


    # Graphique 2
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    xF2 = np.arange(len(Peak_species))
    ax2.bar(xF2 - bar_width, np.sum(F2_red, axis=0), width=bar_width, label="Reduced", color='green')
    ax2.bar(xF2, np.sum(F2_OptimA, axis=0), width=bar_width, label="Optim A", color='red')
    ax2.bar(xF2 + bar_width, np.sum(F2_OptimB, axis=0), width=bar_width, label="Optim B", color='blue')
    ax2.set_xticks(xF2)
    NT_F2= [rf"$Y_{{{x}}}$" for x in Peak_species]
    ax2.set_xticklabels(NT_F2, rotation=45)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$F_2(Y_i)$")
    ax2.set_ylim([1e0, 1e3])
    ax2.legend(loc='upper center', ncol=3, fontsize=10)
    fig2.savefig(f"F2_plot_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", dpi=300, bbox_inches='tight')

    ####################
    
    # Définition du seuil
    seuil = 1e0

    # Filtrer les indices où au moins une valeur dépasse le seuil
    filtered_indices = [
        i for i in range(len(Peak_species)) 
        if max(np.sum(F2_red, axis=0)[i], np.sum(F2_OptimA, axis=0)[i], np.sum(F2_OptimB, axis=0)[i]) >= seuil
    ]

    # Mise à jour des listes filtrées
    Peak_species_filtered = [Peak_species[i] for i in filtered_indices]
    F2_red_filtered = np.sum(F2_red, axis=0)[filtered_indices]
    F2_OptimA_filtered = np.sum(F2_OptimA, axis=0)[filtered_indices]
    F2_OptimB_filtered = np.sum(F2_OptimB, axis=0)[filtered_indices]

    # Mise à jour des indices pour le plot
    xF2 = np.arange(len(Peak_species_filtered))

    # Création du graphe filtré
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    ax2.bar(xF2 - bar_width, F2_red_filtered, width=bar_width, label="Reduced", color='green')
    ax2.bar(xF2, F2_OptimA_filtered, width=bar_width, label="Optim A", color='red')
    ax2.bar(xF2 + bar_width, F2_OptimB_filtered, width=bar_width, label="Optim B", color='blue')

    # Configuration des axes et légendes
    ax2.set_xticks(xF2)
    ax2.set_xticklabels(Peak_species_filtered, rotation=45)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$F_2(Y_i)$")
    ax2.set_ylim([1e0, 1e3])
    ax2.legend(loc='upper center', ncol=3, fontsize=10)
    plt.rcParams.update({'font.size': 18}) 
    # Enregistrement de la figure
    fig2.savefig(f"F2_Filtered_plot_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", dpi=300, bbox_inches='tight')


    # Créer une figure unique
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))

    # Graphique 3 et Graphique 4 fusionnés en un seul plot
    xF3 = np.arange(len(["T"]))
    # Fusionner les barres dans un seul plot
    ax3.bar(xF3 - bar_width, sum(F4_red), width=bar_width, color='green')
    ax3.bar(xF3, sum(F4_OptimA), width=bar_width, color='red')
    ax3.bar(xF3 + bar_width, sum(F4_OptimB), width=bar_width, color='blue')
    ax3.set_xticks(xF3)
    ax3.set_xticklabels(["T"], rotation=45)
    ax3.set_yscale("log")
    ax3.set_ylabel(r"$F_3(T)$")
    ax3.set_ylim([1e-3, 1e3])
    # Ajouter une seule légende
    handles = [
        plt.Line2D([0], [0], color='green', lw=4, label="Reduced"),
        plt.Line2D([0], [0], color='red', lw=4, label="Optim A"),
        plt.Line2D([0], [0], color='blue', lw=4, label="Optim B")
    ]
    ax3.legend(handles=handles, loc='upper right', ncol=3, fontsize=10)
    plt.rcParams.update({'font.size': 18}) 
    fig3.savefig(f"F3_plot_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", dpi=300, bbox_inches='tight')

    fig4, ax4 = plt.subplots(figsize=(fig_width, fig_height))
    xF4 = np.arange(len(["IDT"]))   # Décalage pour les IDT
    ax4.bar(xF4 - bar_width, sum(F3_red), width=bar_width, color='green')
    ax4.bar(xF4, sum(F3_OptimA), width=bar_width, color='red')
    ax4.bar(xF4 + bar_width, sum(F3_OptimB), width=bar_width, color='blue')

    # Ajuster les ticks
    ax4.set_xticks(xF4)
    ax4.set_xticklabels(["IDT"], rotation=0)

    # Log scale et autres paramètres
    ax4.set_yscale("log")
    ax4.set_ylabel(r"$F_4(IDT)$")
    ax4.set_ylim([1e-3, 1e3])

    # Ajouter une seule légende
    handles = [
        plt.Line2D([0], [0], color='green', lw=4, label="Reduced"),
        plt.Line2D([0], [0], color='red', lw=4, label="Optim A"),
        plt.Line2D([0], [0], color='blue', lw=4, label="Optim B")
    ]
    ax4.legend(handles=handles, loc='upper center', ncol=3, fontsize=10)
    plt.rcParams.update({'font.size': 18}) 
    # Enregistrer la figure fusionnée
    fig4.savefig(f"F4_plot_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", dpi=300, bbox_inches='tight')

        
    ### Plot Error Max of a selected species 
    plt.figure()
    plt.rcParams.update({'font.size': 18}) 
    spec_F1 = "N2O"
    ind_err_F1 = Intergrate_Species.index(spec_F1)
    
    
    plt.plot(np.array(F1_red)[:,ind_err_F1],color="green",label="Reduced")
    plt.plot(np.array(F1_OptimA)[:,ind_err_F1],color="red",label="Optim A")
    plt.plot(np.array(F1_OptimB)[:,ind_err_F1],color="blue",label="Optim B")
    plt.legend()
    plt.xlabel("cases")
    plt.ylabel("F1(N2O)")
    
    plt.savefig(f"F1_Error_cases_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png") #==> Case n°4
    
    ind_case_F1 =np.argmax(np.array(F1_OptimB)[:,ind_err_F1])
    
    fig, ax1 = plt.subplots()
    plt.rcParams.update({'font.size': 14}) 

    # Tracer les courbes de l'axe principal
    ax1.plot(Commun_time[ind_case_F1], Interp_Y_det[ind_case_F1][Targets.index(spec_F1)], "k", label="Detailed")
    ax1.plot(Commun_time[ind_case_F1], Interp_Y_red[ind_case_F1][Targets.index(spec_F1)], "g", label="Reduced")
    ax1.plot(Commun_time[ind_case_F1], Interp_Y_OptimA[ind_case_F1][Targets.index(spec_F1)], "r", label="Optim A")
    ax1.plot(Commun_time[ind_case_F1], Interp_Y_OptimB[ind_case_F1][Targets.index(spec_F1)], "b", label="Optim B")
    # ax1.set_xlim(min(Commun_time[ind_case_F1]), 0.002)
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel(r"$Y_{N_2O}$")
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Récupérer les erreurs associées
    err_red = np.array(F1_red)[ind_case_F1, ind_err_F1]
    err_OptimA = np.array(F1_OptimA)[ind_case_F1, ind_err_F1]
    err_OptimB = np.array(F1_OptimB)[ind_case_F1, ind_err_F1]

    # Définir les erreurs avec leurs couleurs associées
    errors = {
        r"$F_1^{Reduced}(Y_{N_2O})$": (np.array(F1_red)[ind_case_F1, ind_err_F1], "green"),
        r"$F_1^{Optim A}(Y_{N_2O})$": (np.array(F1_OptimA)[ind_case_F1, ind_err_F1], "red"),
        r"$F_1^{OptimB}(Y_{N_2O})$": (np.array(F1_OptimB)[ind_case_F1, ind_err_F1], "blue"),
    }

    # Position de base pour afficher les erreurs (coordonnées relatives)
    x_pos, y_pos = 0.5, 0.5 
    line_spacing = 0.1  # Espacement vertical entre les lignes

    # Ajouter chaque ligne avec sa couleur
    for i, (label, (value, color)) in enumerate(errors.items()):
        ax1.annotate(f"{label}: {value:.2e}", xy=(x_pos, y_pos - i * line_spacing),
                    xycoords="axes fraction", fontsize=12, color=color)


    # Placer la légende au-dessus du plot en une seule ligne
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12, frameon=False)

    # Sauvegarde de la figure
    plt.savefig(f"Case_N2O_Error_Max_PMO_ST_{Time_shift}_LOG_{log}_SC_{scaler}.png", bbox_inches="tight")
