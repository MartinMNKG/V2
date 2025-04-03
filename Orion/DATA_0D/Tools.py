import numpy as np
import cantera as ct
import os
import sys
import time
import warnings
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle

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

def Sim0D_launch(t_gas,gas_eq,fuel1,fuel2,oxidizer,all_case,idx_target,idx_non_target,dt,tmax)  : 
   
    
    time_list = [] 
    temp_list =[]
    Y_target_list = [] 
    Y_non_target_list = [] 
 
    for case in all_case :
        # pressure, temperature, equivalence_ratio,mixture = case

        # fuel_mix = f'{fuel1}:{mixture}, {fuel2}:{1-mixture}'
        # gas.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
        # gas.TP = temperature,pressure
        print(case)
        t_list, tem_list, y_list = Sim0D(t_gas,gas_eq,fuel1,fuel2,oxidizer,case,dt,tmax)
        _Y_target_list = y_list[:,np.array(idx_target)]
        _Y_non_target_list = y_list[:,np.array(idx_non_target)]
        time_list.append(t_list)
        temp_list.append(tem_list)
        Y_target_list.append(_Y_target_list)
        Y_non_target_list.append(_Y_non_target_list)
    
    
    return time_list,temp_list,Y_target_list,Y_non_target_list

def Change_detailed(time_list,temp_list,Y_target_list,Y_non_target_list,case,target,non_target,param):
     
    info_shift = param[0]
    info_interp = param[1]
    info_log = param[2]
    info_scal = param[3]
    
    ai_delay = Calc_ai_delay(time_list,temp_list,case)
    
    if info_shift == "shift":
        shift_time_list = shift(time_list, temp_list,case)
        shift_time_list_scale = shift_scale(time_list, temp_list,case)
    
    if info_interp =="interpol" : 
        Y_target_list = interp_Y(shift_time_list,Y_target_list,shift_time_list_scale,case,target)
            
        Y_non_target_list = interp_Y(shift_time_list,Y_non_target_list,shift_time_list_scale,case,non_target)   
        
        temp_list = interp_T(shift_time_list,temp_list,shift_time_list_scale,case)
    
    if info_log == "log":
        Y_target_list = np.log(Y_target_list)
        Y_non_target_list = np.log(Y_non_target_list)
        
    if info_scal =="scaler" :
        
        Y_target_list,Y_scaler_Targets = standard(Y_target_list,case,target)
        Y_non_target_list,Y_scaler_Non_Targets = standard(Y_non_target_list,case,non_target)
        temp_list,Temp_scaler = standard_T(temp_list,case)
        
    elif info_scal =="noscaler" : 
        Y_scaler_Targets= None 
        Y_scaler_Non_Targets= None
        Temp_scaler = None
    
    return shift_time_list_scale,temp_list,Y_target_list,Y_non_target_list,Y_scaler_Targets,Y_scaler_Non_Targets, ai_delay,Temp_scaler 

def Change_Reduced(time_list,temp_list,Y_target_list,Y_non_target_list,case,target,non_target,Time_Det,Scal_Target,Scal_Non_Target,Temp_scaler,param):
     
    info_shift = param[0]
    info_interp = param[1]
    info_log =param[2]
    info_scal = param[3]
    
    ai_delay = Calc_ai_delay(time_list,temp_list,case)
    
    if info_shift == "shift":
        time_list = shift(time_list, temp_list,case)
    
    if info_interp =="interpol" : 
        Y_target_list = interp_Y(time_list,Y_target_list,Time_Det,case,target)
            
        Y_non_target_list = interp_Y(time_list,Y_non_target_list,Time_Det,case,non_target)   
        
        temp_list = interp_T(time_list,temp_list,Time_Det,case)
        
    if info_log == "log":
        Y_target_list = np.log(Y_target_list)
        Y_non_target_list = np.log(Y_non_target_list)
        
    if info_scal =="scaler" :
        
        Y_target_list = standard_rdc(Y_target_list,case,target,Scal_Target)
        Y_non_target_list = standard_rdc(Y_non_target_list,case,non_target,Scal_Non_Target)
        temp_list =standard_rdc_T(temp_list,case,Temp_scaler)
    
    
    return time_list,temp_list,Y_target_list,Y_non_target_list , ai_delay 


def Sim0D(t_gas,gas_eq,fuel1,fuel2,oxidizer,case,dt,tmax,tol=1e-9):
    
    pressure, temperature, equivalence_ratio,mixture = case

    fuel_mix = f'{fuel1}:{mixture}, {fuel2}:{1-mixture}'
    t_gas.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
    t_gas.TP = temperature,pressure
    
    r = ct.IdealGasConstPressureReactor(t_gas)
    sim = ct.ReactorNet([r])
    # sim.max_steps=100000
    t_list = [0]
    temp_list = [t_gas.T]
    y_list = []
    y_list.append(r.Y)
    
    n_iter = 0 
    time = 0
    equil_bool = False 
    
    gas_eq.TP = temperature,pressure
    gas_eq.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
    gas_eq.equilibrate("HP")
    state_equil = np.append(gas_eq.X, gas_eq.T)
    
    while time <tmax:
        time +=dt
        sim.advance(time)
        # time = sim.step()
        t_list.append(time)
        temp_list.append(r.T)
        y_list.append(r.Y)
        state_current = np.append(r.thermo.X, r.T)
        residual = (
            100
            * np.linalg.norm(state_equil - state_current, ord=np.inf)
            / np.linalg.norm(state_equil, ord=np.inf)
        )
        n_iter +=1 
        # print(residual)
        # if residual < tol :  #if time >= tmax :
        #     return t_list, temp_list, np.array(y_list)
        if len(temp_list) > 200 : 
            if np.abs(temp_list[-1]  - temp_list[-2]) < tol :
                return t_list, temp_list, np.array(y_list)
    
    return t_list, temp_list, np.array(y_list)
    
def Sim0D_old(t_gas,gas_eq,fuel1,fuel2,oxidizer,case,dt,tmax,tol=0.3):
    
    
    pressure, temperature, equivalence_ratio,mixture = case

    fuel_mix = f'{fuel1}:{mixture}, {fuel2}:{1-mixture}'
    t_gas.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
    t_gas.TP = temperature,pressure
    
    r = ct.IdealGasConstPressureReactor(t_gas)
    sim = ct.ReactorNet([r])
    # sim.max_steps=100000
    t_list = [0]
    temp_list = [t_gas.T]
    y_list = []
    y_list.append(r.Y)
    
    n_iter = 0 
    time = 0
    equil_bool = False 
    
    gas_eq.TP = temperature,pressure
    gas_eq.set_equivalence_ratio(equivalence_ratio,fuel_mix,oxidizer)
    gas_eq.equilibrate("HP")
    state_equil = np.append(gas_eq.X, gas_eq.T)
    states = ct.SolutionArray(t_gas, extra=["t"])
    
    while time <tmax:
        # time +=dt
        time = sim.step()
        t_list.append(time)
        temp_list.append(r.T)
        y_list.append(r.Y)
        state_current = np.append(r.thermo.X, r.T)
        states.append(r.thermo.state, t=time)
        residual = (
            100
            * np.linalg.norm(state_equil - state_current, ord=np.inf)
            / np.linalg.norm(state_equil, ord=np.inf)
        )
        n_iter +=1 
        # print(residual)
        if residual < tol and time*10/100>tmax : #if time >= tmax :
            print(t_gas.n_species)
            if t_gas.n_species == 33 : 
                states.save(f"ref.csv",overwrite=True)   
            else : 
                states.save(f"red.csv",overwrite=True) 
            return t_list, temp_list, np.array(y_list)
    
    return t_list, temp_list, np.array(y_list)
 

def generate_test_cases_bifuel(temp_range, pressure_range, second_param,mixture):
    test_cases = list(itertools.product(pressure_range, temp_range, second_param,mixture))

    # Convertir les pressions en Pascals (Pa) car Cantera utilise les Pascals
    test_cases = [(p * 101325, T, second,mixture) for p, T, second,mixture in test_cases]
    
    return test_cases

def spcs_name_idx(gas, spcs_ipt):
    spcs_idx = []
    for m in range(len(spcs_ipt)):
        for k in range(gas.n_species):
            if gas.species_name(k) == spcs_ipt[m]:
                spcs_idx.append(k)
    return spcs_idx

def non_target(gas,targets) : 
    list_non_target = []
    for i in gas.species_names : 
        if i not in targets : 
            list_non_target.append(i)
    return list_non_target

def shift_scale(time_l, temp_l,case) :
    shift_time =[]
    for c in range(len(case)) : 
        loc_time = np.array(time_l[c])
        
        loc_temp = np.array(temp_l[c])
        
        grad = np.gradient(loc_temp,loc_time)
        idx_grad = np.argmax(grad)
        shift_grid = loc_time - loc_time[idx_grad]
        
        shift_time.append(np.linspace(min(shift_grid),max(shift_grid),100))
        
    return shift_time  

def shift(time_l, temp_l,case) :
    shift_time =[]
    for c in range(len(case)) : 
        loc_time = np.array(time_l[c])
        
        loc_temp = np.array(temp_l[c])
        
        grad = np.gradient(loc_temp,loc_time)
        idx_grad = np.argmax(grad)
        shift_grid = loc_time - loc_time[idx_grad]
        
        shift_time.append(shift_grid)
        
    return shift_time  

def interp_Y(data_grid,data_value,commun_grid,case,list_spec):
    output = []
    for c in range(len(case)): 
        loc_grid = np.array(data_grid[c])
        loc_data_value= np.array(data_value[c])
        loc_commun_grid = np.array(commun_grid[c])
        
        output_loc =[]   
        for spec in range(len(list_spec)) :
            loc_loc_data_value = loc_data_value[:,spec]
            int_func = interp1d(loc_grid,loc_loc_data_value,fill_value="extrapolate")
            output_loc.append(int_func(loc_commun_grid))
            
        output.append(output_loc)
    return output

def interp_T(data_grid,data_value,commun_grid,case) :
    output= [] 
    for c in range(len(case)) : 
        loc_grid = np.array(data_grid[c])
        loc_data_value = np.array(data_value[c])
        loc_commun_grid = np.array(commun_grid[c])
        int_func = interp1d(loc_grid,loc_data_value,fill_value="extrapolate")
        output.append(int_func(loc_commun_grid))
    return output

def standard(data_value,case,list_spec): 
    output = []
    scaler = []
    for c in range(len(case)) : 
        loc_data_value= np.array(data_value[c])
        output_loc =[]
        scaler_loc=[]
        
        for spec in range(len(list_spec)) :
            scl = MinMaxScaler()

            loc_loc_data_value = loc_data_value[spec]
            scl.fit(loc_loc_data_value)
            output_loc.append(scl.transform(loc_loc_data_value))
            scaler_loc.append(scl)
        output.append(output_loc)
        scaler.append(scaler_loc)
    return output,scaler
 
def standard_T(data_value,case) :
    output = []
    scaler = []
    for c in range(len(data_value)) :
        loc_data_value = np.array(data_value[c])
        scl = MinMaxScaler() 
        scl.fit(loc_data_value)
        output.append(scl.transform(loc_data_value))
        scaler.append(scl)
    return output,scaler

def standard_rdc_T(data_value,case,list_scaler) : 
    output = [] 
    for c in range(len(case)) : 
        loc_data_value = np.array(data_value[c])
        scl = list_scaler[c]
        output.append(scl.transform(loc_data_value))
    return output

def standard_rdc(data_value,case,list_spec,list_scaler) : 
    output = [] 
    for c in range(len(case)) : 
        loc_data_value = np.array(data_value[c])
        loc_scal = list_scaler[c]
        output_loc =[]
        
        for spec in range(len(list_spec)): 
            loc_loc_data_value = loc_data_value[spec]
            scl = loc_scal[spec]
            output_loc.append(scl.transform(loc_loc_data_value))
        
        output.append(output_loc)
    return output 

def calc_delay_diff(t, temp):
    t_arr = np.array(t)
    temp_arr = np.array(temp)
    d_temp = temp_arr[1:] - temp_arr[:-1]
    d_t = t_arr[1:] - t_arr[:-1]
    d_temp_d_t = d_temp / d_t
    delay_idx = np.argmax(np.abs(d_temp_d_t))
    delay = (t[delay_idx] + t[delay_idx + 1]) / 2
    return delay 
    
                  
def Calc_ai_delay(time,temp,case) : 
    output = [] 
    for c in range(len(case)) : 
        loc_time = time[c]
        loc_temp = temp[c]
        diff_temp = np.diff(loc_temp)/np.diff(loc_time)
        ign_pos = np.argmax(diff_temp)
        output.append(loc_time[ign_pos])
    return output
  
def load_ref(pop_size,ngen,mutpb) : 
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/Time_det.pkl", "rb") as f:
        Time_det = pickle.load(f)
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/Y_Target_det.pkl", "rb") as f:
        Y_Target_det = pickle.load(f)
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/Y_Non_Target_det.pkl", "rb") as f:
        Y_Non_Target_det = pickle.load(f)
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/Temp_det.pkl", "rb") as f:
        Temp_det = pickle.load(f)
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/AI_delay_det.pkl", "rb") as f:
        AI_delay_det = pickle.load(f)
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/Scaler_Target_det.pkl", "rb") as f:
        Scaler_Target_det = pickle.load(f)
    with open(f"./Ouput_mpi_pop{pop_size}_gen{ngen}_mut{mutpb}/Scaler_Non_Target_det.pkl", "rb") as f:
        Scaler_Non_Target_det = pickle.load(f)
        
    return Time_det, Y_Target_det, Y_Non_Target_det, Temp_det, AI_delay_det, Scaler_Target_det, Scaler_Non_Target_det
                 
def write_yaml(gas, filename):
    t_writer = ct.YamlWriter()
    t_writer.add_solution(gas)
    t_writer.to_file(filename)
    return   
    
def plot(x,y,case):
    for c in range(len(case)) :  
        plt.figure()
        plt.plot(x[c][:],y[c][:])
    pass

def fit_pyoptmec_test(Y_t_red,Y_t_det,time_det,Y_nt_red,Y_nt_det,temp_red,temp_det,ai_det,ai_red,case) : 
    F1_m =[]
    F2_m =[]
    F3_m =[] 
    F4_m = []
    for c in range(len(case)) : 
        # Equation F1_m
        top = np.trapezoid(np.abs(np.array(Y_t_red[c]) - np.array(Y_t_det[c])), np.array(time_det[c]))
        bot = np.trapezoid(np.abs(np.array(Y_t_det[c])), np.array(time_det[c]))
        F1_m.append( (top/bot)**2)

        # Equation F2_m
        top = [(np.max(Y_nt_red[c],axis=1)[j] - np.max(Y_nt_det[c],axis=1)[j]) for j in range(len(np.max(Y_nt_det[c],axis=1))) ]
        bot = [np.max(Y_nt_det[c],axis=1)[j] for j in range(len(np.max(Y_nt_det[c],axis=1))) ]
    
        F2_m.append((np.array(top)/np.array(bot))**2)

        # Equation F3_m
        top = np.trapezoid(np.abs(temp_red[c] - temp_det[c]), time_det[c])
        bot = np.trapezoid(np.abs(temp_det[c]), time_det[c])
        F3_m .append(( top / bot)**2)

        # Equation F4_m
        top = (ai_red[c] - ai_det[c])
        bot= ai_det[c]
        F4_m.append((top/bot)**2)
        
    weight = [1,1,1,1]
    # print(f"F1 = {F1_m}")
    # print(f"F2 = {F2_m}")
    
    print(f"F1 = {np.sum(F1_m)}")
    print(f"F2 = {np.sum(F2_m)}")
    print(f"F3 = {F3_m}")
    print(f"F4 = {F4_m}")
    _err = (
                weight[0] * np.sum(F1_m)
                + weight[1] * np.sum(F2_m)
                + weight[2] * F3_m
                + weight[3] * F4_m
            )
    # print(f"Err : {_err}")

    err = np.sqrt(np.sum(_err))
    print(np.sqrt(np.sum(_err)))
    return err,


def fit_dummy(case, AI_delay_det, AI_delay_red):
    err_ai = []
    
    # Calcul de l'erreur pour chaque élément de case
    for c in range(len(case)):
        err_ai.append(np.abs(np.array(AI_delay_det[c]) - np.array(AI_delay_red[c])))
    
    # Calcul de la fitness
    fitness = sum(err_ai) / len(err_ai)
    
    return fitness,

def fit_OptimSmoke(Y_exp, Y_sim,case):
    fit =[]
    fac = []
    for c in range(len(case)) :
        fit.append(np.sum(np.sum((np.array(Y_exp[c]) -np.array(Y_sim[c]))**2,axis = 1)))
        fac.append(np.shape(Y_exp[c])[1])
    return np.sum(np.array(fit)/np.array(fac)),


def rxns_yaml_arr_list2(t_gas, factor):
    # print(factor)
    species = t_gas.species()
    reactions = t_gas.reactions()
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=reactions
    )
    rxns_orig = _gas.reactions()
    rxns_modd = []
    p = 0
    for k in range(_gas.n_reactions):
        t_rxn = rxns_orig[k]
        str_equ = t_rxn.equation
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        t_dup = t_rxn.duplicate
        str_dup = ""
        if t_dup:
            str_dup = ",\nduplicate: true"
        if type_rxns == "Arrhenius":
            
            rate_a =  factor[p]
            p = p + 1
            rate_b =  factor[p]
            p = p + 1
            rate_e =  factor[p]
            p = p + 1
            str_rate = (
                "{A: "
                + str(rate_a)
                + ", b: "
                + str(rate_b)
                + ", Ea: "
                + str(rate_e)
                + "}"
            )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "rate-constant: "
                + str_rate
                + str_dup
                + "}"
            )
        elif type_rxns == "pressure-dependent-Arrhenius":
            t_rates = t_rxn.rate.rates
            str_rate = ""
            for m in range(len(t_rates)):
                t_pres = t_rates[m][0] / 101325.0
                t_rate = t_rates[m][1]
                rate_a =  factor[p]
                p = p + 1
                rate_b =  factor[p]
                p = p + 1
                rate_e =  factor[p]
                p = p + 1
                str_rate = str_rate + (
                    "{P: "
                    + str(t_pres)
                    + " atm, A: "
                    + str(rate_a)
                    + ", b: "
                    + str(rate_b)
                    + ", Ea: "
                    + str(rate_e)
                    + "},\n"
                )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: pressure-dependent-Arrhenius"
                + str_dup
                + ",\n"
                + "rate-constants: \n["
                + str_rate
                + "]\n"
                + "}"
            )
        elif type_rxns == "three-body-Arrhenius":
            str_eff = str(t_rxn.third_body.efficiencies)
            str_eff = str_eff.replace("'", "")
            rate_a =  factor[p]
            p = p + 1
            rate_b =  factor[p]
            p = p + 1
            rate_e =  factor[p]
            p = p + 1
            str_rate = "[" + str(rate_a) + "," + str(rate_b) + "," + str(rate_e) + "]"
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: three-body,\n"
                + "rate-constant: "
                + str_rate
                + ",\n"
                + "efficiencies: "
                + str_eff
                + str_dup
                + "}"
            )
            # print(idx)
            # print(str_cti)
            # return ct.Reaction.fromCti(str_cti)
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            # str_type_falloff = t_rxn.falloff.falloff_type
            array_para_falloff = t_rxn.rate.falloff_coeffs
            str_eff = str(t_rxn.third_body.efficiencies)
            str_eff = str_eff.replace("'", "")
            lowrate_a =  factor[p]
            p = p + 1
            lowrate_b =  factor[p]
            p = p + 1
            lowrate_e =  factor[p]
            p = p + 1
            highrate_a =  factor[p]
            p = p + 1
            highrate_b =  factor[p]
            p = p + 1
            highrate_e =  factor[p]
            p = p + 1
            str_lowrate = (
                "{A: "
                + str(lowrate_a)
                + ", b: "
                + str(lowrate_b)
                + ", Ea: "
                + str(lowrate_e)
                + "}"
            )
            str_highrate = (
                "{A: "
                + str(highrate_a)
                + ", b: "
                + str(highrate_b)
                + ", Ea: "
                + str(highrate_e)
                + "}"
            )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: falloff,\n"
                + "low-P-rate-constant: "
                + str_lowrate
                + ",\n"
                + "high-P-rate-constant: "
                + str_highrate
                + ",\n"
            )
            if type_rxns == "falloff-Troe":
                if len(array_para_falloff) == 4:
                    str_cti = (
                        str_cti
                        + "Troe: {"
                        + "A: "
                        + str(array_para_falloff[0])
                        + ", T3: "
                        + str(array_para_falloff[1])
                        + ", T1: "
                        + str(array_para_falloff[2])
                        + ", T2: "
                        + str(array_para_falloff[3])
                        + "},\n"
                    )
                elif len(array_para_falloff) == 3:
                    str_cti = (
                        str_cti
                        + "Troe: {"
                        + "A: "
                        + str(array_para_falloff[0])
                        + ", T3: "
                        + str(array_para_falloff[1])
                        + ", T1: "
                        + str(array_para_falloff[2])
                        + "},\n"
                    )
            str_cti = str_cti + "efficiencies: " + str_eff + str_dup + "}"
            # print(str_cti)
        tt_rxn = ct.Reaction.from_yaml(str_cti, _gas)
        rxns_modd.append(tt_rxn)
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=rxns_modd
    )
    return _gas



def get_factor_dim_ln(t_gas):
    init_value = []
    species = t_gas.species()
    reactions = t_gas.reactions()
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=reactions
    )
    rxns_orig = _gas.reactions()
    p = 0
    for k in range(_gas.n_reactions):
        t_rxn = rxns_orig[k]
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        # print(type_rxns)
        if type_rxns == "Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor
            
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent
            
            p = p + 1
            rate_e = t_rxn.rate.activation_energy
            p = p + 1
            init_value.append(np.log(rate_a))
            init_value.append(rate_b)
            init_value.append(rate_e)
        elif type_rxns == "pressure-dependent-Arrhenius":
            t_rates = t_rxn.rate.rates
            for m in range(len(t_rates)):
                t_pres = t_rates[m][0] / 101325.0
                t_rate = t_rates[m][1]
                rate_a = t_rate.pre_exponential_factor
                p = p + 1
                rate_b = t_rate.temperature_exponent
                p = p + 1
                rate_e = t_rate.activation_energy
                p = p + 1
                init_value.append(np.log(rate_a))
                init_value.append(rate_b)
                init_value.append(rate_e)
        elif type_rxns == "three-body-Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent
            p = p + 1
            rate_e = t_rxn.rate.activation_energy
            p = p + 1
            init_value.append(np.log(rate_a))
            init_value.append(rate_b)
            init_value.append(rate_e)
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            lowrate_a = t_rxn.rate.low_rate.pre_exponential_factor
            p = p + 1
            lowrate_b = t_rxn.rate.low_rate.temperature_exponent
            p = p + 1
            lowrate_e = t_rxn.rate.low_rate.activation_energy
            p = p + 1
            highrate_a = t_rxn.rate.high_rate.pre_exponential_factor
            p = p + 1
            highrate_b = t_rxn.rate.high_rate.temperature_exponent
            p = p + 1
            highrate_e = t_rxn.rate.high_rate.activation_energy
            p = p + 1
            init_value.append(np.log(lowrate_a))
            init_value.append(lowrate_b)
            init_value.append(lowrate_e)
            init_value.append(np.log(highrate_a))
            init_value.append(highrate_b)
            init_value.append(highrate_e)
        else:
            warnings.warn("Unsupported reaction type " + type_rxns + ".")
    return p,init_value



def rxns_yaml_arr_list2_ln(t_gas, factor):
    # print(factor)
    species = t_gas.species()
    reactions = t_gas.reactions()
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=reactions
    )
    rxns_orig = _gas.reactions()
    rxns_modd = []
    p = 0
    for k in range(_gas.n_reactions):
        t_rxn = rxns_orig[k]
        str_equ = t_rxn.equation
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        t_dup = t_rxn.duplicate
        str_dup = ""
        if t_dup:
            str_dup = ",\nduplicate: true"
        if type_rxns == "Arrhenius":
            
            rate_a =  np.exp(factor[p])
            p = p + 1
            rate_b =  factor[p]
            p = p + 1
            rate_e =  factor[p]
            p = p + 1
            str_rate = (
                "{A: "
                + str(rate_a)
                + ", b: "
                + str(rate_b)
                + ", Ea: "
                + str(rate_e)
                + "}"
            )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "rate-constant: "
                + str_rate
                + str_dup
                + "}"
            )
        elif type_rxns == "pressure-dependent-Arrhenius":
            t_rates = t_rxn.rate.rates
            str_rate = ""
            for m in range(len(t_rates)):
                t_pres = t_rates[m][0] / 101325.0
                t_rate = t_rates[m][1]
                rate_a =  np.exp(factor[p])
                p = p + 1
                rate_b =  factor[p]
                p = p + 1
                rate_e =  factor[p]
                p = p + 1
                str_rate = str_rate + (
                    "{P: "
                    + str(t_pres)
                    + " atm, A: "
                    + str(rate_a)
                    + ", b: "
                    + str(rate_b)
                    + ", Ea: "
                    + str(rate_e)
                    + "},\n"
                )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: pressure-dependent-Arrhenius"
                + str_dup
                + ",\n"
                + "rate-constants: \n["
                + str_rate
                + "]\n"
                + "}"
            )
        elif type_rxns == "three-body-Arrhenius":
            str_eff = str(t_rxn.third_body.efficiencies)
            str_eff = str_eff.replace("'", "")
            
            rate_a =  np.exp(factor[p])
            p = p + 1
            rate_b =  factor[p]
            p = p + 1
            rate_e =  factor[p]
            p = p + 1
            str_rate = "[" + str(rate_a) + "," + str(rate_b) + "," + str(rate_e) + "]"
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: three-body,\n"
                + "rate-constant: "
                + str_rate
                + ",\n"
                + "efficiencies: "
                + str_eff
                + str_dup
                + "}"
            )
            # print(idx)
            # print(str_cti)
            # return ct.Reaction.fromCti(str_cti)
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            # str_type_falloff = t_rxn.falloff.falloff_type
            array_para_falloff = t_rxn.rate.falloff_coeffs
            str_eff = str(t_rxn.third_body.efficiencies)
            str_eff = str_eff.replace("'", "")
            
            lowrate_a =  np.exp(factor[p])
            p = p + 1
            lowrate_b =  factor[p]
            p = p + 1
            lowrate_e =  factor[p]
            p = p + 1
            highrate_a =  np.exp(factor[p])
            p = p + 1
            highrate_b =  factor[p]
            p = p + 1
            highrate_e =  factor[p]
            p = p + 1
            str_lowrate = (
                "{A: "
                + str(lowrate_a)
                + ", b: "
                + str(lowrate_b)
                + ", Ea: "
                + str(lowrate_e)
                + "}"
            )
            str_highrate = (
                "{A: "
                + str(highrate_a)
                + ", b: "
                + str(highrate_b)
                + ", Ea: "
                + str(highrate_e)
                + "}"
            )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: falloff,\n"
                + "low-P-rate-constant: "
                + str_lowrate
                + ",\n"
                + "high-P-rate-constant: "
                + str_highrate
                + ",\n"
            )
            if type_rxns == "falloff-Troe":
                if len(array_para_falloff) == 4:
                    str_cti = (
                        str_cti
                        + "Troe: {"
                        + "A: "
                        + str(array_para_falloff[0])
                        + ", T3: "
                        + str(array_para_falloff[1])
                        + ", T1: "
                        + str(array_para_falloff[2])
                        + ", T2: "
                        + str(array_para_falloff[3])
                        + "},\n"
                    )
                elif len(array_para_falloff) == 3:
                    str_cti = (
                        str_cti
                        + "Troe: {"
                        + "A: "
                        + str(array_para_falloff[0])
                        + ", T3: "
                        + str(array_para_falloff[1])
                        + ", T1: "
                        + str(array_para_falloff[2])
                        + "},\n"
                    )
            str_cti = str_cti + "efficiencies: " + str_eff + str_dup + "}"
            # print(str_cti)
        tt_rxn = ct.Reaction.from_yaml(str_cti, _gas)
        rxns_modd.append(tt_rxn)
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=rxns_modd
    )
    return _gas
