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

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# cd Python
# .venv/Scripts/activate

LengthOptim = 6

# TemperatureMin = 18 * np.ones(LengthOptim)
IrradianceInertia = 15*np.array(
    [0, 0, 0, 0,0,0,0, 1, 2, 2,2,2,2,2, 2, 2, 2,2,2,1,1,0, 0, 0]
)
weightEnv = 0.5
weightCost = 0.5
# weightEnv = float(sys.argv[2])
# weightCost = float(sys.argv[3])
# Read the line of data from the input
# line = sys.argv[1]
# #Decode the line to UTF-8 and print it
# lineDecoded = line#.decode("UTF-8")
# values = [float(i) for i in lineDecoded.split(',')]    # <<< this should work  # added a list   # comprehension to       # convert values to integers
# TemperatureMin = np.array(values)

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

GridPrice = np.array(
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
# GridPrice = 100*np.ones(LengthOptim)

EnergyDeficitCost = 10
flexCost = 10
Priority = np.ones((LengthOptim))

BPrice = weightEnv * CO2emissions + weightCost * GridPrice
EmaxDaily = 100000
BPrice = BPrice[0:LengthOptim]


ProductionSeller = EmaxDaily * np.ones(LengthOptim)
ProductionSeller = 2*np.array(
    [0, 0, 0, 0,0,0,0, 1, 2, 2,2,2,2,3, 3, 2, 2,2,2,1,1,0, 0, 0]
)

DemandRequired = 0*np.ones(LengthOptim)


E_cycle1 = np.array([3, 4, 2])
E_cycle2 = np.array([5])
#   f =np.concatenate([BPrice[:,0],-SPrice[:,0],np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),-confort*np.ones((LengthOptim))]);
# Vector x = [     Demand supplied from ; available production ; demand needs       ; |sum of real demand  - sum of demand needs|;  |Demand needs - real demand| to assess flexibility cost;  Power heater;  Temperature ; alphathermal1] We take temperature as a constraint, not something to optimise. Could be changed.

f = np.r_[
    BPrice, #Demand supplied from the grid/gas engine
    # 0*BPrice, #Demand supplied from PV
    # np.zeros([LengthOptim]),
    # np.zeros([LengthOptim]), #energy needs
    # EnergyDeficitCost,
    # flexCost * Priority,
    # BPrice, #Heat/cooling Demand supplied from the grid/gas engine 
    # 0*BPrice, #Heat/cooling Demand supplied from PV 
    # np.zeros([LengthOptim]), #Temperature
    # np.zeros([LengthOptim]), # alpha1
    np.zeros([LengthOptim-2]), #start of appliance
    np.zeros([LengthOptim-1]), # second timeslot of use of the appliance
    np.zeros([LengthOptim]), # third timeslot of use of the appliance
    np.zeros([LengthOptim]), # Start of second appliance timeslot of use of the appliance
]
#pour un appareil dont le temps de travail est de 3h, on crée 3 variables entières 
# (qui seront les booléens, comme tu disais) de 22 éléments (pour pas surcharger, cf plus bas). 
# Une pour le temps auquel l'appareil démarre (donc ça peut pas démarrer à 23h sinon ça finit pas,
#  d'où le 22 éléments), une  pour le temps auquel l'appareil fait son deuxième time step de travail
#  et une pour dire à quel moment il fait son dernier timestep

#  Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
Index_binary =np.array(list(range(len(f)-3*LengthOptim+2+1+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
# Define the cutoff index
cutoff_index = len(f) - LengthOptim * 4 + 3

# Create the array with 0s up to the cutoff index and 1s after
integrality = np.zeros(len(f))
integrality[cutoff_index:] = 1
# Aeq1 = np.c_[
#     np.zeros((LengthOptim, LengthOptim)),
#     # np.eye(LengthOptim),
#     np.zeros((LengthOptim, LengthOptim)),
#     # np.zeros([LengthOptim, 1]),
#     # np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
# ]
# Aeq2 = np.c_[
#     np.eye(LengthOptim),
#     np.eye(LengthOptim),
#     # np.zeros((LengthOptim, LengthOptim)),
#     # np.eye(LengthOptim),
#     # np.zeros((LengthOptim, 1)),
#     # np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
# ] # Energy from grid + energy from PV = DemandRequired
# Aeq3 = np.c_[
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     # np.zeros((LengthOptim, LengthOptim)),
#     # np.zeros((LengthOptim, LengthOptim)),
#     # np.zeros((LengthOptim, 1)),
#     # np.zeros((LengthOptim, LengthOptim)),
#     alphaThermal2 * np.eye(LengthOptim),
#     alphaThermal2 * np.eye(LengthOptim),
#     -(1 - alphaThermal1)
#     * (np.tri(LengthOptim, LengthOptim, -1) - np.tri(LengthOptim, LengthOptim, -2))
#     + np.eye(LengthOptim),
#     -np.eye(LengthOptim) * (Temperature_Ext)+ np.eye(LengthOptim)*np.r_[np.array([TemperatureInit]), np.zeros(LengthOptim - 1)]
# ] #thermal physical law

# zeros(LengthOptim)                      zeros(LengthOptim)                  zeros(LengthOptim)       zeros(LengthOptim,1)           zeros(LengthOptim)            -alphaThermal2*eye(LengthOptim)  diag(-(1-alphaThermal1)*ones(LengthOptim-1,1),-1)+eye(LengthOptim)

# Aeq3=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),gamma*np.eye(LengthOptim),-alpha*np.eye((LengthOptim),k=-1)+np.eye(LengthOptim),],axis=1)
# Aeq4=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.eye(LengthOptim)],axis=1)
# Aeq = np.concatenate([Aeq1, Aeq2, Aeq3], axis=0)
# Aeq = np.concatenate([Aeq2, Aeq3], axis=0)
# print(Aeq)
# Beq1 = ProductionSeller
# Beq2 = DemandRequired
# Beq3 = (1 - 0) * np.r_[np.array([TemperatureInit]), np.zeros(LengthOptim - 1)]+0.1*IrradianceInertia/100
# Beq = np.concatenate([ Beq2, Beq3], axis=0)
# Beq = np.concatenate([Beq1, Beq2, Beq3], axis=0)

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

PVmax = 0


Aeq1a = np.c_[ #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))

    np.zeros((LengthOptim, LengthOptim)),
    # np.zeros((LengthOptim, LengthOptim)),
    First_Slot_Appliance, #appliance
    -1/2*Second_Slot_Appliance,#appliance
    -1/2*Third_Slot_Appliance,#appliance
    np.zeros((LengthOptim, LengthOptim))
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
#Une avec 24 lignes qui dit (comme tu as proposé) 
# que la variable 1 à t = 1/2*("variable 2 à t+1" + "variable 3 à t+2")
Beq1a = np.zeros((LengthOptim,1))




Aeq2a = np.c_[ #to ensure the appliance will run and only once
    np.zeros((1, LengthOptim)),
    # np.zeros((1, LengthOptim)),
    np.ones((1, LengthOptim-2)),#appliance
    np.zeros((1, LengthOptim-1)),#appliance
    np.zeros((1, LengthOptim)),#appliance
    np.zeros((1, LengthOptim)),#appliance
] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
Beq2a = [[1]]
Aeq2b = np.c_[ #to ensure the appliance will run and only once
    np.zeros((1, LengthOptim)),
    # np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim-2)),#appliance
    np.zeros((1, LengthOptim-1)),#appliance
    np.zeros((1, LengthOptim)),#appliance
    np.ones((1, LengthOptim)),#appliance
] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
Beq2b = [[1]]
# Aeq3 = np.c_[ #to ensure the appliance will run and only once
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim-2)),#appliance
#     np.ones((1, LengthOptim-1)),#appliance
#     np.zeros((1, LengthOptim)),#appliance
# ] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
# Beq3 = [[1]]
# Aeq4 = np.c_[ #to ensure the appliance will run and only once
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim)),
#     np.zeros((1, LengthOptim-2)),#appliance
#     np.zeros((1, LengthOptim-1)),#appliance
#     np.ones((1, LengthOptim)),#appliance
# ] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
# Beq4 =[[1]]


Aeq5 = np.c_[ #Demand = demand from the cycles

    np.eye(LengthOptim, LengthOptim),
    # np.zeros((LengthOptim, LengthOptim)),
    -E_cycle1[0]*np.eye(LengthOptim, LengthOptim-2), #appliance
    -E_cycle1[1]*np.eye(LengthOptim, LengthOptim-1),#appliance
    -E_cycle1[2]*np.eye(LengthOptim, LengthOptim),#appliance
    -E_cycle2[0]*np.eye(LengthOptim, LengthOptim),#appliance
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
Beq5 = np.zeros((LengthOptim,1))



# A1 = np.c_[
#     np.zeros((LengthOptim, LengthOptim)),
#     np.eye((LengthOptim)),
#     # np.zeros((1, LengthOptim)),
#     # -1 * np.ones((1, LengthOptim)),
#     # np.array([0]),
#     # np.zeros((1, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.eye((LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim))
# ] #demand for supply from PV + demand for heating from PV < PV production
# # A1 = np.concatenate([np.ones((LengthOptim)), np.zeros((LengthOptim)), -1*np.ones((LengthOptim)), 0,             np.zeros((LengthOptim)),  np.zeros((LengthOptim)), np.zeros((LengthOptim)),],axis=1)
# #  ones(1,LengthOptim)    zeros(1,LengthOptim)        -1*ones(1,LengthOptim) 0                    zeros(1,LengthOptim)  zeros(1,LengthOptim)  zeros(1,LengthOptim)  ;  %sum demand <= sum need
# # A2 = np.c_[
# #     np.eye((LengthOptim)),
# #     # -1 * np.eye((LengthOptim)),
# #     np.zeros((LengthOptim, LengthOptim)),
# #     np.zeros(LengthOptim),
# #     np.zeros((LengthOptim, LengthOptim)),
# #     np.zeros((LengthOptim, LengthOptim)),
# #     np.zeros((LengthOptim, LengthOptim)),
# #     np.zeros((LengthOptim, LengthOptim)),
# # ]  # demand < production



# # A = np.concatenate([A1, A2, A3, A4, A5, A6], axis=0)
# A = np.concatenate([A1], axis=0)

# b = np.r_[
#     ProductionSeller
#     # np.zeros((LengthOptim)),
#     # np.zeros([LengthOptim]),
#     # np.zeros([LengthOptim]),
#     # np.array([0]),
#     # np.array([0]),
# ]
lb = np.r_[
    np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
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
    np.zeros((LengthOptim)),

]
ub = np.r_[
    EmaxDaily * np.ones((LengthOptim)),
    # PVmax * np.ones((LengthOptim)),
    # EmaxDaily * np.ones((LengthOptim)),
    # EmaxDaily * np.ones((LengthOptim)),
    # np.array([EmaxDaily]),
    # EmaxDaily * np.ones((LengthOptim)),
    # Pmaxchauf * np.ones((LengthOptim)),
    # Pmaxchauf * np.ones((LengthOptim)),
    # TemperatureMax,
    # (alphaThermal1max) * np.ones(LengthOptim),
    np.ones((LengthOptim-2)),
    np.ones((LengthOptim-1)),
    np.ones((LengthOptim)),
    np.ones((LengthOptim)),
]

# model=gp.Model()
# x=model.addMVar(f.shape[0],ub=ub,lb=lb)
# x.obj=f
# model.addConstr(A@x<=b)
# model.addConstr(Aeq@x==Beq)
# model.optimize()
# solution=x.X

############### Scipy ################

# A = np.concatenate([A1, A2, A3, A4, A5, A6, Aeq, np.eye(f.shape[0])], axis=0)
A = np.concatenate([Aeq1a, Aeq2a,Aeq2b, Aeq5, np.eye(f.shape[0])], axis=0)
bl = np.concatenate([Beq1a, Beq2a,Beq2b,  Beq5, lb.reshape((lb.shape[0], 1))], axis=0)
bu = np.concatenate([Beq1a, Beq2a, Beq2b,  Beq5, ub.reshape((ub.shape[0], 1))], axis=0)

constraints = LinearConstraint(A, bl.flatten(), bu.flatten())
# integrality = Index_binary

res = milp(c=f, constraints=constraints, integrality=integrality)
solution = res.x
print("Solution:", solution)

DemandTotale = solution[0:LengthOptim]
Appliance_start = solution[f.shape[0]-LengthOptim*4+3:f.shape[0]-LengthOptim*3+1]
Appliance_cycle2 = solution[f.shape[0]-LengthOptim*3+1:f.shape[0]-LengthOptim*2]
Appliance_cycle3 = solution[f.shape[0]-LengthOptim*2:f.shape[0]-LengthOptim+1]
Appliance2_start = solution[f.shape[0]-LengthOptim*1:f.shape[0]-0*LengthOptim+1]

Appliance_start = np.pad(Appliance_start, (0, 24 - Appliance_start.size), mode='constant')
Appliance_cycle2 = np.pad(Appliance_cycle2, (0, 24 - Appliance_cycle2.size), mode='constant')
Appliance_boolean = Appliance_start+Appliance_cycle2+Appliance_cycle3
# Convert elements: values <= 0.1 become 0, close to 1 become 1
Appliance_boolean = np.where(Appliance_boolean > 0.1, 1, 0)
# Find the first non-zero index in arr2
non_zero_indices = np.nonzero(Appliance_boolean)[0]
start_index = non_zero_indices[0] if non_zero_indices.size > 0 else None

# Initialize result array with zeros, the same length as arr2
Appliance_Load = np.zeros_like(Appliance_boolean)

# Perform the element-wise multiplication if a non-zero start index exists
if start_index is not None and start_index + E_cycle1.size <= Appliance_boolean.size:
    Appliance_Load[start_index:start_index + E_cycle1.size] = E_cycle1 * Appliance_boolean[start_index:start_index + E_cycle1.size]

print(Appliance_Load)