import json
import numpy as np
import itertools

# Fonction pour charger les paramètres d'une flamme spécifique à partir du fichier JSON
def charger_parametres(fichier, type_de_flamme):
    with open(fichier, 'r') as f:
        donnees = json.load(f)
    
    # Vérifier si le type de flamme est dans le fichier
    if type_de_flamme not in donnees["flammes"]:
        raise ValueError(f"Type de flamme '{type_de_flamme}' non trouvé dans le fichier.")
    
    # Charger les données spécifiques à la flamme
    flamme = donnees["flammes"][type_de_flamme]

    # Charger la liste ou la plage de températures
    if flamme["temperature"]["mode"] == "range":
        min_temp = flamme["temperature"]["valeur_min"]
        max_temp = flamme["temperature"]["valeur_max"]
        nombre_max = flamme["temperature"]["nombre_max"]
        temperatures = np.linspace(min_temp, max_temp, nombre_max).tolist()
    else:
        temperatures = flamme["temperature"]["valeurs"]

    #Charger la liste ou la plage de pression 
    if flamme["pressure"]["mode"] == "range":
        min_pressure = flamme["pressure"]["valeur_min"]
        max_pressure = flamme["pressure"]["valeur_max"]
        nombre_max = flamme["pressure"]["nombre_max"]
        pressures = np.linspace(min_pressure, max_pressure, nombre_max).tolist()
    else:
        pressures = flamme["pressure"]["valeurs"]

    # Charger soit l'équivalence ratio, soit le strain en fonction du type de flamme
    if type_de_flamme == "counterflow":
        if flamme["strain"]["mode"] == "range":
            min_strain = flamme["strain"]["valeur_min"]
            max_strain = flamme["strain"]["valeur_max"]
            nombre_max = flamme["strain"]["nombre_max"]
            strains = np.linspace(min_strain, max_strain, nombre_max).tolist()
        else:
            strains = flamme["strain"]["valeurs"]
        return temperatures,pressures, strains
    else:
        if flamme["equivalence_ratio"]["mode"] == "range":
            min_ratio = flamme["equivalence_ratio"]["valeur_min"]
            max_ratio = flamme["equivalence_ratio"]["valeur_max"]
            nombre_max = flamme["equivalence_ratio"]["nombre_max"]
            equivalence_ratios = np.linspace(min_ratio, max_ratio, nombre_max).tolist()
        else:
            equivalence_ratios = flamme["equivalence_ratio"]["valeurs"]
            
        if flamme["mixture"] == "Pure" : 
            return temperatures, pressures, equivalence_ratios
        elif flamme["mixture"] == "Bifuel" : 
            mixture = flamme["mixture"]["valeurs"]
            return temperatures, pressures, equivalence_ratios,mixture
        
    
    
def generate_test_cases(temp_range, pressure_range, second_param):

    test_cases = list(itertools.product(pressure_range, temp_range, second_param))

    # Convertir les pressions en Pascals (Pa) car Cantera utilise les Pascals
    test_cases = [(p * 101325, T, second) for p, T, second in test_cases]
    
    return test_cases


def generate_test_cases_bifuel(temperature,pressure,eq_ratio,mixture):

    test_cases = list(itertools.product(pressure, temperature, eq_ratio,mixture))

    # Convertir les pressions en Pascals (Pa) car Cantera utilise les Pascals
    test_cases = [(p * 101325, T, second) for p, T, second in test_cases]
    
    return test_cases


