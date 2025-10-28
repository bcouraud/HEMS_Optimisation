import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import datetime as dt
import math
from scipy.io import loadmat
from pandas import read_csv
import pandas as pd
# import gurobipy as gp
import sys
import scipy as sc
from scipy.optimize import LinearConstraint
from scipy.optimize import milp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# cd Python
# .venv/Scripts/activate

LengthOptim = 96


# One day in minutes
minutes_per_day = LengthOptim
time = np.arange(minutes_per_day)  # 0 to 1439

# --- Solar Production Profile ---
# --- Base solar profile (ideal clear-sky) ---
# Sunrise at 6:00 (360), sunset at 18:00 (1080)
solar = np.zeros(minutes_per_day)
daylight = (time >= 360/1440*minutes_per_day) & (time <= 1080/1440*minutes_per_day)
solar[daylight] = np.sin(np.pi * (time[daylight] - 360/1440*minutes_per_day) / (1080/1440*minutes_per_day - 360/1440*minutes_per_day))
solar = np.clip(solar, 0, None)
solar *= 5  # peak 5 kW

# --- Add small random fluctuations (sensor noise, atmospheric variation) ---
noise = np.random.normal(0, 0.1, minutes_per_day)  # mean 0, std 0.1 kW
solar_noisy = solar + noise
solar_noisy = np.clip(solar_noisy, 0, None)

# --- Add "cloud events" (temporary dips) ---
n_clouds = np.random.randint(3, 6)  # number of clouds during the day
for _ in range(n_clouds):
    start = np.random.randint(360/1440*minutes_per_day, 1080/1440*minutes_per_day)  # between sunrise and sunset
    duration = np.random.randint(int(5/1440*minutes_per_day), int(30/1440*minutes_per_day))   # 5–30 minutes
    depth = np.random.uniform(0.3, 0.8)   # fraction of reduction
    end = min(start + duration, 1080/1440*minutes_per_day)
    solar_noisy[start:end] *= depth

# --- Smooth slightly to avoid sharp edges ---
window = 5  # minutes
solar_realistic = np.convolve(solar_noisy, np.ones(window)/window, mode='same')
solar = solar_realistic

# --- Household Consumption Profile ---
# Model: higher morning (6–9h) and evening (18–22h) usage, baseline otherwise
base_load = 0.3 + 0.1 * np.random.randn(minutes_per_day)  # small random variation
morning_peak = ((time >= 360/1440*96) & (time <= 540/1440*96)) * (0.5 + 0.1 * np.random.randn(minutes_per_day))   # 6–9h
evening_peak = ((time >= 1080/1440*96) & (time <= 1320/1440*96)) * (0.6 + 0.1 * np.random.randn(minutes_per_day)) # 18–22h
consumption = base_load + morning_peak + evening_peak
consumption = np.clip(consumption, 0.1, None)  # avoid negative consumption






weightEnv = 0.5
weightCost = 0.5
EVpower = 4

CO2emissions = np.array(
    [
        175.3832496,
        153.1666087,
        140.7375224,
        178.1145482,
        184.0737135,
        160.3737913,
        121.2004819,
        124.9227102,
        155.6844156,
        179.8172943,
        181.9825352,
        186.7212864,
        175.3832496,
        153.1666087,
        140.7375224,
        178.1145482,
        184.0737135,
        160.3737913,
        121.2004819,
        124.9227102,
        155.6844156,
        179.8172943,
        181.9825352,
        186.7212864,
    ]
)

GridImportPrice = np.array(
    [
        113.4,
        110.4,
        106.5,
        117.4,
        150.8,
        161,
        167.1,
        148.7,
        319.4,
        345.8,
        156.2,
        118.4,
        113.4,
        110.4,
        106.5,
        117.4,
        150.8,
        161,
        167.1,
        148.7,
        319.4,
        345.8,
        156.2,
        118.4,
    ]
)
n_in = len(GridImportPrice)
n_out = LengthOptim  # or 96, etc.

# Stepwise (zero-order hold) interpolation:
repeat_factor = n_out // n_in
GridImportPrice_step = np.repeat(GridImportPrice, repeat_factor)
CO2emissions_step = np.repeat(CO2emissions, repeat_factor)
# Adjust for rounding if n_out is not perfectly divisible
if len(GridImportPrice_step) < n_out:
    GridImportPrice_step = np.append(
        GridImportPrice_step,
        [GridImportPrice[-1]] * (n_out - len(GridImportPrice_step))
    )
    CO2emissions_step = np.append(
        CO2emissions_step,
        [CO2emissions[-1]] * (n_out - len(CO2emissions_step))
    )
elif len(GridImportPrice_step) > n_out:
    GridImportPrice_step = GridImportPrice_step[:n_out]
    CO2emissions_step = CO2emissions_step[:n_out]
GridImportPrice = GridImportPrice_step
CO2emissions = CO2emissions_step
# GridImportPrice = 100*np.ones(LengthOptim)
GridExportPrice = 0.5*GridImportPrice
# GridExportPrice[13:15] = 300
BPrice = GridImportPrice # weightEnv * CO2emissions + weightCost * GridImportPrice
EmaxDaily = 100000
Appliance_Availability = np.ones(LengthOptim)
BPrice = BPrice[0:LengthOptim]
GridExportPrice = GridExportPrice[0:LengthOptim]
ProductionSeller = EmaxDaily * np.ones(LengthOptim)
ProductionSeller = solar #np.array(
#     [0, 0, 0, 0,0,0,0, 1, 2, 2.5 ,2,2.5,2,3, 3.5, 3, 2.5,2,1.5,1.2,1,0, 0, 0]
# )
ProductionSeller = ProductionSeller[0:LengthOptim]
ProductionSeller = ProductionSeller.reshape(LengthOptim,1)

DemandRequired = consumption # np.ones(LengthOptim)
EV_Energy = 12
presence = np.ones((LengthOptim))
presence[int(8/24*minutes_per_day):int(188/24*minutes_per_day)] = 0
E_cycle = np.array([5, 4, 4.5, 4])# np.array([5, 4, 2])

f = np.r_[
    BPrice, #Demand supplied from the grid/gas engine
    0*BPrice, #Demand supplied from PV
    -GridExportPrice, #exports to the grid
    # np.zeros([LengthOptim]),
    # np.zeros([LengthOptim]), #energy needs
    # EnergyDeficitCost,
    # flexCost * Priority,
    # BPrice, #Heat/cooling Demand supplied from the grid/gas engine 
    # 0*BPrice, #Heat/cooling Demand supplied from PV 
    # np.zeros([LengthOptim]), #Temperature
    # np.zeros([LengthOptim]), # alpha1
    np.zeros([LengthOptim]), # Xi_1, non binary, required to enable Export = 0  if Import > 0 and vice versa (Egridin -Xi_1 = Demand - PV. Xi_1 is here to enable the left hand side to be negative)
    np.zeros([LengthOptim]), # Xi_2, non binary, required to enable  Export = 0  if Import > 0 and vice versa (Egridout -Xi_2 = PV-Demand  Xi_2 is here to enable the left hand side to be negative
    np.zeros([LengthOptim]), # binary1 so Export = 0  if Import > 0 and vice versa
    np.zeros([LengthOptim-2]), #binary start of appliance
    np.zeros([LengthOptim-1]), # binary second timeslot of use of the appliance
    np.zeros([LengthOptim]), # binary third timeslot of use of the appliance
    np.zeros([LengthOptim]) # Power drawn for EV
]
#pour un appareil dont le temps de travail est de 3h, on crée 3 variables entières 
# (qui seront les booléens, comme tu disais) de 22 éléments (pour pas surcharger, cf plus bas). 
# Une pour le temps auquel l'appareil démarre (donc ça peut pas démarrer à 23h sinon ça finit pas,
#  d'où le 22 éléments), une  pour le temps auquel l'appareil fait son deuxième time step de travail
#  et une pour dire à quel moment il fait son dernier timestep

#  Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
Index_binary =np.array(list(range(len(f)-5*LengthOptim+2+1+1,len(f)*2+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
# Define the cutoff index
cutoff_index = len(f) - LengthOptim * 5 + 3

# Create the array with 0s up to the cutoff index and 1s after
integrality = np.zeros(len(f))
integrality[cutoff_index:len(f) - LengthOptim] = 1

# Create a matrix with zeros everywhere
First_Slot_Appliance = np.zeros((LengthOptim, LengthOptim-2))
# Add ones to the superdiagonal (above the main diagonal)
np.fill_diagonal(First_Slot_Appliance[:-1, 0:], 1)
# Create a matrix with zeros everywhere
Second_Slot_Appliance = np.zeros((LengthOptim, LengthOptim-1))
# Add ones to the superdiagonal (above the main diagonal)
np.fill_diagonal(Second_Slot_Appliance[:-1, 1:], 1)
# Display the matrix
print(Second_Slot_Appliance)
# Create a matrix with zeros everywhere
Third_Slot_Appliance = np.zeros((LengthOptim, LengthOptim))
# Add ones to the superdiagonal (above the main diagonal)
np.fill_diagonal(Third_Slot_Appliance[:-2, 2:], 1)


Aeq1 = np.c_[ #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    First_Slot_Appliance, #appliance
    -1/2*Second_Slot_Appliance,#appliance
    -1/2*Third_Slot_Appliance,#appliance
    np.zeros((LengthOptim, LengthOptim)), #EV

] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
#Une avec 24 lignes qui dit que la variable 1 à t = 1/2*("variable 2 à t+1" + "variable 3 à t+2")
Beq1 = np.zeros((LengthOptim,1))

Aeq2 = np.c_[ #to ensure the appliance will run and only once
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.ones((1, LengthOptim-2)),#appliance
    np.zeros((1, LengthOptim-1)),#appliance
    np.zeros((1, LengthOptim)),#appliance
    np.zeros((1, LengthOptim)),#EV
] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
Beq2 = [[1]]
# Aeq3 = np.c_[ #to ensure the appliance will run and only once
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim-2)),#appliance
#     np.zeros((1, LengthOptim-1)),#appliance
#     np.zeros((1, LengthOptim)),#appliance
# ] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
# Beq3 = [[0]]
# Aeq4 = np.c_[ #to ensure the appliance will run and only once
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim-2)),#appliance
#     np.zeros((1, LengthOptim-1)),#appliance
#     np.zeros((1, LengthOptim)),#appliance
# ] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
# Beq4 =[[0]]


Aeq3 = np.c_[ #Demand from grid + demand from PV = demand from the cycles
    np.eye(LengthOptim, LengthOptim),
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    -E_cycle[0]*np.eye(LengthOptim, LengthOptim-2), #appliance
    -E_cycle[1]*np.eye(LengthOptim, LengthOptim-1),#appliance
    -E_cycle[2]*np.eye(LengthOptim, LengthOptim),#appliance
    -1* np.eye(LengthOptim, LengthOptim) #EV
] #to ensure the appliance will be supplied by grid or pv
Beq3 = np.zeros((LengthOptim,1)) + DemandRequired.reshape(LengthOptim,1)

# From Now, this is for PV integration to avoid import and export in the same time.
Aeq4 = np.c_[ #PV production = export PV to grid + demand supplied by PV
    np.zeros((LengthOptim, LengthOptim)),
    np.eye(LengthOptim, LengthOptim),
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim-2)), #appliance
    np.zeros((LengthOptim, LengthOptim-1)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#EV
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
Beq4 = ProductionSeller 


Aeq5 = np.c_[ #Egridin - Xi_1 = Demand - PV
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    -np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    -E_cycle[0]*np.eye(LengthOptim, LengthOptim-2), #appliance
    -E_cycle[1]*np.eye(LengthOptim, LengthOptim-1),#appliance
    -E_cycle[2]*np.eye(LengthOptim, LengthOptim),#appliance
    -1*np.eye(LengthOptim, LengthOptim)
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
Beq5 = DemandRequired - ProductionSeller.flatten()

Aeq6 = np.c_[ #Egridout - Xi_2 = PV - Demand 
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    -np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    E_cycle[0]*np.eye(LengthOptim, LengthOptim-2), #appliance
    E_cycle[1]*np.eye(LengthOptim, LengthOptim-1),#appliance
    E_cycle[2]*np.eye(LengthOptim, LengthOptim),#appliance
    1*np.eye(LengthOptim, LengthOptim) #EV
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
Beq6 = ProductionSeller.flatten()-DemandRequired


Aeq7 = np.c_[ #Egridout - Xi_2 = PV - Demand 
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim-2)),
    np.zeros((1, LengthOptim-1)),
    np.zeros((1, LengthOptim)),
    1*np.ones((1, LengthOptim)) #EV
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
Beq7 = [[EV_Energy]]


A1 = np.c_[ # Export < (1-alph1)Emax
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)), #
    np.zeros((LengthOptim, LengthOptim)), #
    EmaxDaily*np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim-2)), #appliance
    np.zeros((LengthOptim, LengthOptim-1)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#appliance
    np.zeros((LengthOptim, LengthOptim))#EV
]  #to ensure the export will not happen when there is import
B1 = EmaxDaily*np.ones((LengthOptim,1))

A2 = np.c_[ # # Import < (alph1)Emax
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),   
    np.zeros((LengthOptim, LengthOptim)), #
    np.zeros((LengthOptim, LengthOptim)), #
    -EmaxDaily*np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim-2)), #appliance
    np.zeros((LengthOptim, LengthOptim-1)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#appliance
    np.zeros((LengthOptim, LengthOptim))#EV
] #to ensure the export will not happen when there is import
B2 = np.zeros((LengthOptim,1))
A3 = np.c_[ # Export < (1-alph1)Emax
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),   
    np.zeros((LengthOptim, LengthOptim)), #
    np.eye(LengthOptim, LengthOptim),
    -EmaxDaily*np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim-2)), #appliance
    np.zeros((LengthOptim, LengthOptim-1)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#appliance
    np.zeros((LengthOptim, LengthOptim))#appliance
]  #to ensure the export will not happen when there is import
B3 = np.zeros((LengthOptim,1))

A4 = np.c_[ # # Import < (alph1)Emax
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),   
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim)), #
    EmaxDaily*np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim-2)), #appliance
    np.zeros((LengthOptim, LengthOptim-1)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#appliance
    np.zeros((LengthOptim, LengthOptim))#EV
] #to ensure the export will not happen when there is import
B4 = EmaxDaily*np.ones((LengthOptim,1))

lb = np.r_[
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
    # np.array([0]),
    # np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
    # TemperatureMin,
    # (alphaThermal1min) * np.ones(LengthOptim),
    np.zeros((LengthOptim-2)),
    np.zeros((LengthOptim-1)),
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)) #EV
]
ub = np.r_[
    EmaxDaily * np.ones((LengthOptim)),
    ProductionSeller.flatten(),
    ProductionSeller.flatten(),
    EmaxDaily * np.ones((LengthOptim)),
    EmaxDaily * np.ones((LengthOptim)),
    np.ones((LengthOptim)),
    # EmaxDaily * np.ones((LengthOptim)),
    # EmaxDaily * np.ones((LengthOptim)),
    # np.array([EmaxDaily]),
    # EmaxDaily * np.ones((LengthOptim)),
    # Pmaxchauf * np.ones((LengthOptim)),
    # Pmaxchauf * np.ones((LengthOptim)),
    # TemperatureMax,
    # (alphaThermal1max) * np.ones(LengthOptim),
    Appliance_Availability[0:LengthOptim-2],#np.ones((LengthOptim-2)),
    np.ones((LengthOptim-1)),
    np.ones((LengthOptim)),
    EVpower*presence
]

# model=gp.Model()
# x=model.addMVar(f.shape[0],ub=ub,lb=lb)
# x.obj=f
# model.addConstr(A@x<=b)
# model.addConstr(Aeq@x==Beq)
# model.optimize()
# solution=x.X

############### Scipy ################

# A = np.concatenate([A1, A2, A3, A4, Aeq1, Aeq2, Aeq3, Aeq4, Aeq5, Aeq6, Aeq7,Aeq8, np.eye(f.shape[0])], axis=0)
# bl = np.concatenate([-EmaxDaily*np.ones((LengthOptim,1)), -EmaxDaily*np.ones((LengthOptim,1)),  -EmaxDaily*np.ones((LengthOptim,1)),  -EmaxDaily*np.ones((LengthOptim,1)), Beq1, Beq2, Beq3, Beq4, Beq5, Beq6, Beq7.reshape(LengthOptim,1), Beq8.reshape(LengthOptim,1), lb.reshape((lb.shape[0], 1))], axis=0)
# bu = np.concatenate([B1, B2, B3, B4, Beq1, Beq2, Beq3, Beq4, Beq5, Beq6,  Beq7.reshape(LengthOptim,1), Beq8.reshape(LengthOptim,1), ub.reshape((ub.shape[0], 1))], axis=0)

A = np.concatenate([A1, A2, A3, A4, Aeq1, Aeq2,  Aeq3, Aeq4, Aeq5,Aeq6, Aeq7,np.eye(f.shape[0])], axis=0)
bl = np.concatenate([-EmaxDaily*np.ones((LengthOptim,1)), -EmaxDaily*np.ones((LengthOptim,1)),  -EmaxDaily*np.ones((LengthOptim,1)),  -EmaxDaily*np.ones((LengthOptim,1)), Beq1, Beq2,  Beq3, Beq4, Beq5.reshape(LengthOptim,1), Beq6.reshape(LengthOptim,1), Beq7, lb.reshape((lb.shape[0], 1))], axis=0)
bu = np.concatenate([B1, B2, B3, B4, Beq1, Beq2,  Beq3, Beq4,  Beq5.reshape(LengthOptim,1), Beq6.reshape(LengthOptim,1), Beq7, ub.reshape((ub.shape[0], 1))], axis=0)


constraints = LinearConstraint(A, bl.flatten(), bu.flatten())

res = milp(c=f, constraints=constraints, integrality=integrality)
solution = res.x
print("Solution:", solution)

DemandfromGrid= solution[0:LengthOptim]
DemandfromPV= solution[LengthOptim:LengthOptim*2]
ExportPV= solution[LengthOptim*2:LengthOptim*3]
Appliance_start = solution[f.shape[0]-LengthOptim*4+3:f.shape[0]-LengthOptim*3+1]
Appliance_cycle2 = solution[f.shape[0]-LengthOptim*3+1:f.shape[0]-LengthOptim*2]
Appliance_cycle3 = solution[f.shape[0]-LengthOptim*2:f.shape[0]-LengthOptim]
Appliance_start = np.pad(Appliance_start, (0, LengthOptim - Appliance_start.size), mode='constant')
Appliance_cycle2 = np.pad(Appliance_cycle2, (0, LengthOptim - Appliance_cycle2.size), mode='constant')
Appliance_boolean = Appliance_start+Appliance_cycle2+Appliance_cycle3
EV_consumption = solution[f.shape[0]-LengthOptim:f.shape[0]+1]

# Convert elements: values <= 0.1 become 0, close to 1 become 1
Appliance_boolean = np.where(Appliance_boolean > 0.1, 1, 0)
# Find the first non-zero index in arr2
non_zero_indices = np.nonzero(Appliance_boolean)[0]
start_index = non_zero_indices[0] if non_zero_indices.size > 0 else None

# Initialize result array with zeros, the same length as arr2
Appliance_Load = np.zeros_like(Appliance_boolean)

# Perform the element-wise multiplication if a non-zero start index exists
if start_index is not None and start_index + E_cycle.size <= Appliance_boolean.size:
    Appliance_Load[start_index:start_index + E_cycle.size] = E_cycle * Appliance_boolean[start_index:start_index + E_cycle.size]



Today=dt.datetime(2025,10,2,00,00,00)


timeArray=[Today + dt.timedelta(minutes=1440/minutes_per_day*x) for x in range(0, LengthOptim)]



# Create subplots
fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

# --- COLORS ---
colors = {
    "Production": "rgba(255, 165, 0, 0.3)",     # orange, semi-transparent
    "Demand": "rgba(0, 128, 255, 0.3)",         # blue
    "Appliance": "rgba(0, 200, 100, 0.3)",      # green
    "Export PV": "rgba(252, 144, 3, 0.5)",        # red
    "Demand from PV": "rgba(255, 255, 0, 0.2)", # yellow
    "Demand from Grid": "rgba(128, 0, 128, 0.3)", # purple
    "EV consumption": "rgba(0, 0, 0, 0.3)",     # black/gray
    "Buying Price": "rgba(0, 128, 255, 0.3)",
    "Selling Price": "rgba(255, 0, 0, 0.3)"
}

# --- Plot 1: Power ---
fig.add_trace(go.Scatter(x=timeArray, y=ProductionSeller.flatten(), mode='lines',
                         name='Production', line=dict(color=colors["Production"]),
                         fill='tozeroy', fillcolor=colors["Production"]), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=timeArray, y=DemandRequired, mode='lines',
                         name='Demand', line=dict(color=colors["Demand"]),
                         fill='tozeroy', fillcolor=colors["Demand"]), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=timeArray, y=Appliance_Load, mode='lines',
                         name='Appliance', line=dict(color=colors["Appliance"]),
                         fill='tozeroy', fillcolor=colors["Appliance"]), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=timeArray, y=ExportPV, mode='lines',
                         name='Export PV', line=dict(color=colors["Export PV"]),
                         fill='tozeroy', fillcolor=colors["Export PV"]), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=timeArray, y=DemandfromPV, mode='lines',
                         name='Demand from PV', line=dict(color=colors["Demand from PV"]),
                         fill='tozeroy', fillcolor=colors["Demand from PV"]), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=timeArray, y=DemandfromGrid, mode='lines',
                         name='Demand from Grid', line=dict(color=colors["Demand from Grid"]),
                         fill='tozeroy', fillcolor=colors["Demand from Grid"]), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=timeArray, y=EV_consumption, mode='lines',
                         name='Demand for EV', line=dict(color=colors["EV consumption"]),
                         fill='tozeroy', fillcolor=colors["EV consumption"]), row=1, col=1, secondary_y=False)

# --- Plot 2: Price ---
fig.add_trace(go.Scatter(
    x=timeArray, y=GridImportPrice.flatten(),
    mode='lines', name='Buying Price',
    line=dict(color='red', dash='dash'),  # dashed line
    fill=None  # no fill
), row=1, col=1, secondary_y=True)

fig.add_trace(go.Scatter(
    x=timeArray, y=GridExportPrice.flatten(),
    mode='lines', name='Selling Price',
    line=dict(color='green', dash='dash'),
    fill=None
), row=1, col=1, secondary_y=True)

# --- Layout: Transparent background + styling ---
fig.update_layout(
    height=800,
    width=1600,
    # title_text="Power, Price, and Energy Graphs",
    showlegend=True,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor='rgba(100,100,100,0)',   # Transparent overall background
    plot_bgcolor='rgba(10,10,10,0)',    # Transparent plotting area
    font=dict(color='grey'),        # Optional: white text for dark themes
)
# Row 1: Power
fig.update_yaxes(title_text="Power (kW)", row=1, col=1)

# Row 2: Price
fig.update_yaxes(title_text="Price (€/kWh)", row=2, col=1)
# Optional: tweak grid/axes visibility
fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

fig.show()



# Save each matrix to a separate text file
Aeq = np.concatenate([Aeq1, Aeq2, Aeq5], axis=0)
beq = np.concatenate([Beq1, Beq2,  Beq5], axis=0)
Amax = np.concatenate([np.eye(f.shape[0])], axis=0)
Amin = np.concatenate([np.eye(f.shape[0])], axis=0)
bmax = np.concatenate([ub.reshape((ub.shape[0], 1))], axis=0)
bmin = lb.reshape((lb.shape[0], 1))

def save_matrix(filename, matrix):
    with open(filename, 'w') as f:
        # Start with opening bracket for whole matrix
        f.write("[\n")
        for row in matrix:
            # Each row encapsulated within brackets
            f.write("  [" + ", ".join([f"{num:.2f}" for num in row]) + "]\n")
        # End with closing bracket for whole matrix
        f.write("]\n")
save_matrix('Aeq1.txt', Aeq1)

# Save matrices with desired format
save_matrix('Aeq.txt', Aeq)
save_matrix('Amax.txt', Amax)
save_matrix('Amin.txt', Amin)
save_matrix('beq.txt', beq)
save_matrix('bmax.txt', bmax)
save_matrix('bmin.txt', bmin)


print("demand from Grid: ", DemandfromGrid)
print("demand from PV: ", DemandfromPV)
print("Start_time: ", Appliance_start)
time = [0]


# np.set_printoptions(linewidth=300)
# original_stdout = sys.stdout

# with open('output.txt','w') as file:
#     sys.stdout=file
#     print("f")
#     print(f)
#     print("input:")
#     print(y/1000)
# sys.stdout=original_stdout

# print("Hello ")#+ sys.argv[1] + "or " +  sys.argv[2] +"! temperature = " + Tint)
# sys.stdout.flush()
# plt.figure(1)
# plt.suptitle('Puissances')
# plt.plot(timeenjour,P, timeenjour, D, timeenjour, Powerbat)
# plt.legend(['Production', 'Demand','Battery'])
# plt.xticks(rotation=90)

# plt.figure(2)
# plt.suptitle('Prix')
# plt.plot(timeenjour,BP, timeenjour, SP)
# plt.legend(['Buying Price', 'Selling Price'])
# plt.xticks(rotation=90)

# plt.figure(3)
# plt.suptitle('Puissance de chauffage')
# plt.plot(timeenjour,Pchauff)
# plt.legend(['Pchauffage'])
# plt.xticks(rotation=90)

# plt.figure(4)
# plt.suptitle('Tint et Text')
# plt.plot(timeenjour,T,timeenjour,ConsigneH,timeenjour,ConsigneB,timeenjour,Tin)
# plt.legend(['Temp ext','Temp int max','Temp int min',"Temp int réelle"])
# plt.xticks(rotation=90)
