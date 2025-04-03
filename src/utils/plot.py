import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import re
import itertools

# =====================================================================
#     PLOT ERR RMSE/MAE ==> Comparision
# =====================================================================

def plot_comparision(main_path : str,species_of_interest : list,type_flame) -> None : 
    print("Plot Error")
    
    
    Err_mae = pd.read_csv(os.path.join(main_path,f"Error/{type_flame}/Err/Err_MAE.csv"))
    Err_rmse = pd.read_csv(os.path.join(main_path,f"Error/{type_flame}/Err/Err_RMSE.csv"))
    
    T_init = Err_mae["T_init"].unique()
    P_init = Err_mae["P_init"].unique()
    
    if type_flame =="speedflame" :
        list_spec = ["X_" + col for col in species_of_interest]
        Er_init = Err_mae["ER"].unique()
        combi = list(itertools.product(T_init,P_init,Er_init))
        # =====================================================================
        #     PLOT MAE 
        # =====================================================================
        plt.figure()
        for _t , _p, _er in combi : 
            
            Err_mae_loc = Err_mae[(Err_mae["T_init"]==_t)&(Err_mae["P_init"]==_p)&(Err_mae["ER"]==_er)]
            plt.plot(Err_mae_loc[list_spec].mean(),label = f"ER{_er} T{_t} P{_p}")
            plt.legend()
            plt.grid()
        plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_MAE.png'))
        # =====================================================================
        #     PLOT RMSE ERROR 
        # =====================================================================
        plt.figure()
        for _t , _p, _er in combi :  
            Err_rmse_loc = Err_rmse[(Err_rmse["T_init"]==_t)&(Err_rmse["P_init"]==_p)&(Err_rmse["ER"]==_er)]
            plt.plot(np.sqrt(Err_rmse_loc[list_spec].mean()),label = f"ER{_er} T{_t} P{_p}")
            plt.legend()
            plt.grid()
            
        plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_RMSE.png'))
    
    if type_flame =="0Dreactor" :
        list_spec = ["Y_" + col for col in species_of_interest]
        Er_init = Err_mae["ER"].unique()
        combi = list(itertools.product(T_init,P_init,Er_init))
        # =====================================================================
        #     PLOT MAE 
        # =====================================================================
        plt.figure()
        for _t , _p, _er in combi : 
            
            Err_mae_loc = Err_mae[(Err_mae["T_init"]==_t)&(Err_mae["P_init"]==_p)&(Err_mae["ER"]==_er)]
            plt.plot(Err_mae_loc[list_spec].mean(),label = f"ER{_er} T{_t} P{_p}")
            plt.legend()
            plt.grid()
        plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_MAE.png'))
        # =====================================================================
        #     PLOT RMSE ERROR 
        # =====================================================================
        plt.figure()
        for _t , _p, _er in combi :  
            Err_rmse_loc = Err_rmse[(Err_rmse["T_init"]==_t)&(Err_rmse["P_init"]==_p)&(Err_rmse["ER"]==_er)]
            plt.plot(np.sqrt(Err_rmse_loc[list_spec].mean()),label = f"ER{_er} T{_t} P{_p}")
            plt.legend()
            plt.grid()
        plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_RMSE.png'))
          
    if type_flame =="counterflow" :
        list_spec = ["X_" + col for col in species_of_interest]
        
        St_init = Err_mae["ST"].unique()
        combi = list(itertools.product(T_init,P_init,St_init))
        # =====================================================================
        #     PLOT MAE 
        # =====================================================================
        plt.figure()
        for _t , _p, _st in combi : 
            
            Err_mae_loc = Err_mae[(Err_mae["T_init"]==_t)&(Err_mae["P_init"]==_p)&(Err_mae["ST"]==_st)]
            plt.plot(Err_mae_loc[list_spec].mean(),label = f"ST{_st} T{_t} P{_p}")
            plt.legend()
            plt.grid()
        plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_MAE.png'))
        # =====================================================================
        #     PLOT RMSE ERROR 
        # =====================================================================
        plt.figure()
        for _t , _p, _er in combi :  
            Err_rmse_loc = Err_rmse[(Err_rmse["T_init"]==_t)&(Err_rmse["P_init"]==_p)&(Err_rmse["ST"]==_st)]
            plt.plot(np.sqrt(Err_rmse_loc[list_spec].mean()),label = f"ER{_st} T{_t} P{_p}")
            plt.legend()
            plt.grid()
        plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_RMSE.png'))
    
# =====================================================================
#     PLOT ERR RMSE/MAE ==> Classification
# =====================================================================

def plot_classification(main_path : str,species_of_interest : list,type_flame) -> None : 
    
    if type_flame =="speedflame" or type_flame =="counterflow" : 
        list_spec = ["X_" + col for col in species_of_interest]
    elif type_flame =="0Dreactor" :
        list_spec = ["Y_" + col for col in species_of_interest]
    # =====================================================================
    #     PLOT MAE 
    # =====================================================================
    all_csv_mae = glob.glob(os.path.join(main_path,f"Error/{type_flame}/Err/Err_MAE_*"))
    all_err_mae = pd.DataFrame(columns = list_spec)
    ind = 0
    for csv in all_csv_mae : 
        if ind ==0 :
            match = re.search(r'Err_MAE_(\d+)\.csv$', csv)
            num_max =int(match.group(1))
        data = pd.read_csv(csv,sep=",")
        all_err_mae.loc[ind] = data.mean()
        ind=ind+1
    list_nb_species = list(range(num_max, num_max-ind, -1))
    plt.figure()
    for spec in list_spec : 
        plt.plot(list_nb_species, all_err_mae[spec], label=f"{spec}")
    plt.grid()
    plt.legend()
    plt.xlabel("Nb_species")
    plt.ylabel("Error")
    plt.xticks(list_nb_species)
    plt.title(f"{type_flame}_MAE")
    plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_MAE.png'))
    
    # =====================================================================
    #     PLOT RMSE 
    # =====================================================================
    all_csv_rmse = glob.glob(os.path.join(main_path,f"Error/{type_flame}/Err/Err_RMSE_*"))
    all_err_rmse = pd.DataFrame(columns = list_spec)
    ind = 0
    for csv in all_csv_rmse : 
        if ind ==0 :
            match = re.search(r'Err_RMSE_(\d+)\.csv$', csv)
            num_max =int(match.group(1))
        data = pd.read_csv(csv,sep=",")
        all_err_rmse.loc[ind] = np.sqrt(data.mean())
        ind=ind+1
    list_nb_species = list(range(num_max, num_max-ind, -1))
    plt.figure()
    for spec in list_spec : 
        plt.plot(list_nb_species, all_err_rmse[spec], label=f"{spec}")
    plt.grid()
    plt.legend()
    plt.xlabel("Nb_species")
    plt.ylabel("Error")
    plt.xticks(list_nb_species)
    plt.title(f"{type_flame}_RMSE")
    plt.savefig(os.path.join(main_path,f'Error/{type_flame}/Plot/Err_RMSE.png'))
  

