import pandas as pd 
import cantera as ct
import os 
import re 
import numpy as np
import glob 
import joblib 
from scipy.interpolate import interp1d

class MinMaxScaler:
    def fit(self, x):
        self.min = x.min(0)
        self.max = x.max(0)

    def transform(self, x):
        x = (x - self.min) / (self.max - self.min + 1e-7)
        return x

    def inverse_transform(self, x):
        x = self.min + x * (self.max - self.min + 1e-7)
        return x
# =====================================================================
#     LAUNCH CHANGE GRID (SPACE/TIME) ==> Comparision & Classification
# =====================================================================
def launch_change_grid(main_path : str,lenght : int,type_flame : str, info : str,bifuel : bool,fuel : str) ->None: 
    print("Normalized data")
    #Load ref
    csv_ref = glob.glob(os.path.join(main_path,f"Detailed/*/{type_flame}*"))
    all_data_ref = Load_data_ref(csv_ref,lenght,type_flame,main_path,bifuel,fuel)
    all_data_ref.to_csv(f"{main_path}/Error/{type_flame}/Data/detailed.csv")
    #Load Reduced (only one mecanism)
    if info =="Comparision" : 
        
        csv_red = glob.glob(os.path.join(main_path,f"Reduced/*/{type_flame}*"))
        all_data_reduced = Load_data_reduced(csv_red,all_data_ref,type_flame,main_path,bifuel,fuel)
        all_data_reduced.to_csv(f"{main_path}/Error/{type_flame}/Data/reduced.csv")

    #Load Reduced (all reduced mecanism)
    if info == "Classification": 
        all_path = glob.glob(os.path.join(main_path,f"Reduced/*"))
        for path in all_path : 
            match = re.search(r'/(\d+)S$', path)
            num = match.group(1)
            csv_red = glob.glob(os.path.join(path,f"{type_flame}*"))
            all_data_reduced = Load_data_reduced(csv_red,all_data_ref,type_flame,main_path)
            all_data_reduced.to_csv(f"{main_path}/Error/{type_flame}/Data/reduced_{num}.csv")

# =====================================================================
#     LOAD DATA REF==> Comparision & Classification
# =====================================================================

def Load_data_ref(csv_ref : list,lenght : int,type_flame : str,main_path : str,bifuel : bool, fuel : str) :
    scaler = MinMaxScaler()
    all_data_ref = pd.DataFrame()
    
    
    for csv in csv_ref : 
        New_data = pd.DataFrame() 
        data =pd.read_csv(csv,sep=",")
        #Load info for post prod
        if bifuel == False :
            if type_flame == "speedflame" or type_flame == "0Dreactor": 
                match = re.search(r"(\d+)S_ER([\d.]+)_T([\d.]+)_P([\d.]+).csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ER"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                name_scaler =f"{type_flame}_ER{SecondParam}_T{T_value}_P{P_value}.save"
            if type_flame =="counterflow" :
                match = re.search(r"(\d+)S_ST([\d.]+)_T([\d.]+)_P([\d.]+)\.csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ST"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                name_scaler =f"{type_flame}_ST{SecondParam}_T{T_value}_P{P_value}.save"
        if bifuel == True : 
            if type_flame == "speedflame" or type_flame == "0Dreactor": 
                
                match = re.search(rf"(\d+)S_ER([\d.]+)_T([\d.]+)_P([\d.]+)_{fuel}_([\d.]+).csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ER"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                Mixture_Value = float(match.group(5))
                name_scaler =f"{type_flame}_ER{SecondParam}_T{T_value}_P{P_value}_M{Mixture_Value}.save"
            if type_flame =="counterflow" :
                match = re.search(rf"(\d+)S_ST([\d.]+)_T([\d.]+)_P([\d.]+)_{fuel}_([\d.]+).csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ST"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                Mixture_Value = float(match.group(5))
                name_scaler =f"{type_flame}_ST{SecondParam}_T{T_value}_P{P_value}_M{Mixture_Value}.save"
        
        #Shift Grid (space/time)== commun grid
        if type_flame =="speedflame" or type_flame =="counterflow":
            shift_grid = shift(data["grid"],data["T"])
            commun_grid = np.linspace(min(shift_grid),max(shift_grid),lenght)

            list_species = [col for col in data.columns if col.startswith("X_")] # Only Species 
        
        if type_flame =="0Dreactor" :
            shift_grid = data["t"]-ai_delay(data)
            commun_grid = np.linspace(min(shift_grid),max(shift_grid),lenght)
            list_species = [col for col in data.columns if col.startswith("Y_")] # Only Species 
        
        #Interpolation of the species on the commun grid
        for spec in list_species :
            New_data[spec] = interp(shift_grid,data[spec],commun_grid)
        
        
        scaler.fit(New_data)
        
        # Save Scaler for reduced data scaler
        joblib.dump(scaler,os.path.join(main_path,f"Error/{type_flame}/Scaler/{name_scaler}"))
        New_data = scaler.transform(New_data)
        
        if bifuel == True : 
            New_data["Mixture"] = Mixture_Value
        
        #Add Info 
        if type_flame == "speedflame" or type_flame =="0Dreactor" : 
            New_data["ER"] = SecondParam 
        if type_flame =="counterflow" : 
            New_data["ST"] = SecondParam
            
        New_data["T_init"] = T_value
        New_data["P_init"] = P_value
        New_data["commun_grid"] = commun_grid
        New_data["T"] = data["T"]
        
        if type_flame =="speedflame" or type_flame =="counterflow" : 
            New_data["velocity"] = data["velocity"]
            
        if type_flame =="0Dreactor" : 
            New_data["AI_delay"] = ai_delay(data)
            
        all_data_ref =pd.concat([all_data_ref,New_data],ignore_index=True)
                            
                
    return all_data_ref 

# =====================================================================
#     LOAD DATA REDUCED==> Comparision & Classification
# =====================================================================
def Load_data_reduced(csv_ref : list ,all_data_ref : dict,type_flame: str,main_path:str,bifuel : bool ,fuel :str) :
    
    all_data_red = pd.DataFrame()
    
    for csv in csv_ref : 
        
        New_data = pd.DataFrame() 
        data =pd.read_csv(csv,sep=",")
        # Load info for post prod 
        if bifuel == False : 
            if type_flame == "speedflame"or type_flame == "0Dreactor" : 
                match = re.search(r"(\d+)S_ER([\d.]+)_T([\d.]+)_P([\d.]+)\.csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ER"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                name_scaler =f"{type_flame}_ER{SecondParam}_T{T_value}_P{P_value}.save"
            if type_flame =="counterflow" :
                match = re.search(r"(\d+)S_ST([\d.]+)_T([\d.]+)_P([\d.]+)\.csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ER"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                name_scaler =f"{type_flame}_ST{SecondParam}_T{T_value}_P{P_value}.save"
        
        elif bifuel == True : 
            if type_flame == "speedflame" or type_flame == "0Dreactor": 
                match = re.search(rf"(\d+)S_ER([\d.]+)_T([\d.]+)_P([\d.]+)_{fuel}_([\d.]+).csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ER"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                Mixture_Value = float(match.group(5))
                name_scaler =f"{type_flame}_ER{SecondParam}_T{T_value}_P{P_value}_M{Mixture_Value}.save"
            if type_flame =="counterflow" :
                match = re.search(rf"(\d+)S_ST([\d.]+)_T([\d.]+)_P([\d.]+)_{fuel}_([\d.]+).csv", csv)
                S_value = int(match.group(1))     # Nombre avant "S"
                SecondParam = float(match.group(2))  # Nombre après "ST"
                T_value = float(match.group(3))   # Nombre après "T"
                P_value = float(match.group(4))   # Nombre après "P"
                Mixture_Value = float(match.group(5))
                name_scaler =f"{type_flame}_ST{SecondParam}_T{T_value}_P{P_value}_M{Mixture_Value}.save"
        
        #Shift grid (space/time) & load commun grid 
        if type_flame == "speedflame":
            shift_grid = shift(data["grid"],data["T"])
            data_ref_loc = all_data_ref[(all_data_ref["ER"]==SecondParam)&(all_data_ref["T_init"]==T_value)&(all_data_ref["P_init"]==P_value)]
            list_species = [col for col in data.columns if col.startswith("X_")]
            if bifuel == True : 
                data_ref_loc = all_data_ref[(all_data_ref["Mixture"]==Mixture_Value)&(all_data_ref["ER"]==SecondParam)&(all_data_ref["T_init"]==T_value)&(all_data_ref["P_init"]==P_value)]
        if type_flame =="counterflow" :
            shift_grid = shift(data["grid"],data["T"])
            data_ref_loc = all_data_ref[(all_data_ref["ST"]==SecondParam)&(all_data_ref["T_init"]==T_value)&(all_data_ref["P_init"]==P_value)]
            list_species = [col for col in data.columns if col.startswith("X_")]
            if bifuel == True : 
                data_ref_loc = all_data_ref[(all_data_ref["Mixture"]==Mixture_Value)&(all_data_ref["ST"]==SecondParam)&(all_data_ref["T_init"]==T_value)&(all_data_ref["P_init"]==P_value)]
        
        if type_flame =="0Dreactor" : 
            shift_grid = data["t"]-ai_delay(data)
            data_ref_loc = all_data_ref[(all_data_ref["ER"]==SecondParam)&(all_data_ref["T_init"]==T_value)&(all_data_ref["P_init"]==P_value)]
            list_species = [col for col in data.columns if col.startswith("Y_")]
            if bifuel == True : 
                data_ref_loc = all_data_ref[(all_data_ref["Mixture"]==Mixture_Value)&(all_data_ref["ER"]==SecondParam)&(all_data_ref["T_init"]==T_value)&(all_data_ref["P_init"]==P_value)]
        
        
        #Interpol on commun grid  
        commun_grid = data_ref_loc["commun_grid"]
        for spec in list_species :
            New_data[spec] = interp(shift_grid,data[spec],commun_grid)
            
        #Load scaler and transform 
        scaler = joblib.load(os.path.join(main_path,f"Error/{type_flame}/Scaler/{name_scaler}"))
        New_data = scaler.transform(New_data)
        
        if bifuel == True: 
            New_data["Mixture"] = Mixture_Value
        #Add Info 
        if type_flame == "speedflame" or type_flame =="0Dreactor" : 
            New_data["ER"] = SecondParam 
        if type_flame =="counterflow" : 
            New_data["ST"] = SecondParam
            
        New_data["T_init"] = T_value
        New_data["P_init"] = P_value
        New_data["commun_grid"] = commun_grid
        
        New_data["T"] = data["T"]
        
        if type_flame =="speedflame" or type_flame =="counterflow" : 
            New_data["velocity"] = data["velocity"]
            
        if type_flame =="0Dreactor" : 
            New_data["AI_delay"] = ai_delay(data)
        
            
        all_data_red =pd.concat([all_data_red,New_data],ignore_index=True)
                            
               
    return all_data_red 

# =====================================================================
#     SHIFT GRID (SPACE)==> SpeedFlame & Counterflow
# =====================================================================
def shift(grid: list, T: list) -> list:
    gradient = np.gradient(T, grid)
    indice_gradient = np.argmax(gradient)
    shift_grid = grid - grid.loc[indice_gradient]

    return shift_grid

# =====================================================================
#     SHIFT GRID (TIME)==> 0D REACTOR
# =====================================================================
def ai_delay(data,alpha =0.05):
    time=data["t"]
    Temperature=data["T"]
    T_init = Temperature[0]
    T_max = max(Temperature)
    
    ignition_temp = T_init + alpha * (T_max - T_init)
    
    for i, T in enumerate(Temperature):
        if T >= ignition_temp:
            return time[i]

# =====================================================================
#     INTERPOLATE DATA ON COMMUN GRID
# =====================================================================       
def interp(data_grid : list,data_value : dict,commun_grid : list ):
    
    int_func = interp1d(data_grid, data_value, fill_value="extrapolate")
    output = int_func(commun_grid)
    return output

# =====================================================================
#     LAUNCH ERROR==> Comparision
# =====================================================================
def launch_error_comparision(main_path : str,species_of_interest : list,type_flame :str,bifuel : bool)-> None: 
    print("Calculate Error")
    if type_flame =="speedflame" or type_flame =="counterflow" :
        list_spec = ["X_" + col for col in species_of_interest]
    elif type_flame =="0Dreactor":
        list_spec = ["Y_" + col for col in species_of_interest]
        
    ref_name = os.path.join(main_path,f"Error/{type_flame}/Data/detailed.csv")
    red_name= os.path.join(main_path,f"Error/{type_flame}/Data/reduced.csv")
    
    ref = pd.read_csv(ref_name)
    reduced = pd.read_csv(red_name)
    
    Err_RMSE = pd.DataFrame(columns=list_spec)
    Err_MAE = pd.DataFrame(columns=list_spec)
    for spec in list_spec : 
        Err_RMSE[spec] = calculate_RMSE(ref[spec],reduced[spec])
        Err_MAE[spec] = calculate_MAE(ref[spec],reduced[spec])
        
    if type_flame =="speedflame" or type_flame =="0Dreactor" : 
        Err_MAE["ER"]= ref["ER"]
        Err_RMSE["ER"]= ref["ER"]
        
    if type_flame =="counterflow" : 
        Err_MAE["ST"]= ref["ST"]
        Err_RMSE["ST"]= ref["ST"]
    
    if bifuel == True : 
        Err_MAE["Mixture"]= ref["Mixture"]
        Err_RMSE["Mixture"]= ref["Mixture"]
        
    Err_MAE["T_init"]=ref["T_init"]
    Err_MAE["P_init"]=ref["P_init"]
    
    
    Err_RMSE["T_init"]=ref["T_init"]
    Err_RMSE["P_init"]=ref["P_init"]
    
    
    Err_MAE.to_csv(os.path.join(main_path,f'Error/{type_flame}/Err/Err_MAE.csv'))
    Err_RMSE.to_csv(os.path.join(main_path,f'Error/{type_flame}/Err/Err_RMSE.csv'))

# =====================================================================
#     LAUNCH ERROR==> Classification
# ===================================================================== 
def launch_error_classification(main_path : str,species_of_interest : list,type_flame :str)-> None: 
    
    if type_flame =="speedflame" or type_flame =="counterflow" :
        list_spec = ["X_" + col for col in species_of_interest]
    elif type_flame =="0Dreactor":
        list_spec = ["Y_" + col for col in species_of_interest]
        
    ref_name = os.path.join(main_path,f"Error/{type_flame}/Data/detailed.csv")
    all_csv= glob.glob(os.path.join(main_path,f"Error/{type_flame}/Data/reduced*"))
    
    ref = pd.read_csv(ref_name)
    for red_name in all_csv:
        
        match = re.search(r'reduced_(\d+)\.csv$', red_name)
        num = match.group(1)
        
        reduced = pd.read_csv(red_name)
        
        Err_RMSE = pd.DataFrame(columns=list_spec)
        Err_MAE = pd.DataFrame(columns=list_spec)
        for spec in list_spec : 
            Err_RMSE[spec] = calculate_RMSE(ref[spec],reduced[spec])
            Err_MAE[spec] = calculate_MAE(ref[spec],reduced[spec])
            
        if type_flame =="speedflame" or type_flame =="0Dreactor" : 
            Err_MAE["ER"]= ref["ER"]
            Err_RMSE["ER"]= ref["ER"]
            
        if type_flame =="counterflow" : 
            Err_MAE["ST"]= ref["ST"]
            Err_RMSE["ST"]= ref["ST"]
            
        Err_MAE["T_init"]=ref["T_init"]
        Err_MAE["P_init"]=ref["P_init"]
        
        
        Err_RMSE["T_init"]=ref["T_init"]
        Err_RMSE["P_init"]=ref["P_init"]
        
        
        Err_MAE.to_csv(os.path.join(main_path,f'Error/{type_flame}/Err/Err_MAE_{num}.csv'))
        Err_RMSE.to_csv(os.path.join(main_path,f'Error/{type_flame}/Err/Err_RMSE_{num}.csv'))

# =====================================================================
#     ROOT MEAN SQUARRE ERROR 
# =====================================================================
def calculate_RMSE(ref: dict, value: dict) -> dict:
    return (value - ref) ** 2

# =====================================================================
#     MEAN ABSOLUTE ERROR
# =====================================================================
def calculate_MAE(ref: dict, value: dict) -> dict:
    return (value - ref).abs()