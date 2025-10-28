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

LengthOptim = 24

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
GridPrice = 100*np.ones(LengthOptim)

EnergyDeficitCost = 10
flexCost = 10
Priority = np.ones((LengthOptim))
# alphaThermal2 = 0.021 * 1440 / 120
# alphaThermal1 = 0.065 * 1440 / 120
# alphaThermal2 = 0.015 * 1440 / 120
# alphaThermal1 = 0.075 * 1440 / 120
alphaThermal2 = 0.009 * 1440 / 120
alphaThermal1 = 0.0015 * 1440 / 120
alphaThermal1max =  0.008 * 1440 / 120 # 0.002 * 1440 / 120
alphaThermal1min = alphaThermal1max #0.0015 * 1440 / 120 #alphaThermal1max #

alphaThermal1 = (alphaThermal1min + alphaThermal1max) / 2
BPrice = weightEnv * CO2emissions + weightCost * GridPrice
EmaxDaily = 100000
Pmaxchauf = 100
# TemperatureMin = 15 * np.ones(LengthOptim)
# TemperatureMin = np.array(
#     [15, 15.2, 21.1, 20.3, 11.64, 14.6, 14.5, 11.6, 21.3, 21.1, 20.2, 5]
# )
TemperatureMin = 20.5 * np.ones(LengthOptim)
TemperatureMax = 30 * np.ones(
    LengthOptim
)
# TemperatureMax[10:17] = 27
# Lecture d'un fichier csv pour les données de température extérieure
file_name = "2024-02-23_data_avec_clim.csv"
df = pd.read_csv(file_name, sep=";")
tab_temp = df["T_ext"]
print("Temp ext :", tab_temp)
Temperature_Ext = np.array(tab_temp)
Temperature_Ext= 25*np.ones(LengthOptim)
Temperature_Ext[10:17]=38
# Temperature_Ext[12:14]=12
# Lecture d'un fichier csv pour les données de température intérieure réelle
file_name = "2024-02-23_data_avec_clim.csv"
df = pd.read_csv(file_name, sep=";")
# tab_t_amb = df["T_amb_cvc"]
tab_t_amb = df["T_amb_netatmo"]
print("Temp amb :", tab_t_amb)
Tamb = np.array(tab_t_amb)

ProductionSeller = EmaxDaily * np.ones(LengthOptim)
ProductionSeller = 2*np.array(
    [0, 0, 0, 0,0,0,0, 1, 2, 2,2,2,2,3, 3, 2, 2,2,2,1,1,0, 0, 0]
)

DemandRequired = 0*np.ones(LengthOptim)
TemperatureInit = 27
# TemperatureMin = np.array([TemperatureInit ,20, 20, 0, 20, 23])
# TemperatureMax = 22 * np.ones(
#     LengthOptim
# )  # np.array([28 ,28, 28, 28, 28, 28,28, 28, 28, 28, 28,28])
  # np.array([28 ,28, 28, 28, 28, 28,28, 28, 28, 28, 28,28])

# int(sum(sum(ProductionSeller,DemandRequired)))
#   f =np.concatenate([BPrice[:,0],-SPrice[:,0],np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),-confort*np.ones((LengthOptim))]);
# Vector x = [     Demand supplied from ; available production ; demand needs       ; |sum of real demand  - sum of demand needs|;  |Demand needs - real demand| to assess flexibility cost;  Power heater;  Temperature ; alphathermal1] We take temperature as a constraint, not something to optimise. Could be changed.

f = np.r_[
    BPrice, #Demand supplied from the grid/gas engine
    0*BPrice, #Demand supplied from PV
    # np.zeros([LengthOptim]),
    # np.zeros([LengthOptim]), #energy needs
    # EnergyDeficitCost,
    # flexCost * Priority,
    BPrice, #Heat/cooling Demand supplied from the grid/gas engine 
    0*BPrice, #Heat/cooling Demand supplied from PV 
    np.zeros([LengthOptim]), #Temperature
    np.zeros([LengthOptim]), # alpha1
]

#  Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];

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
Aeq2 = np.c_[
    np.eye(LengthOptim),
    np.eye(LengthOptim),
    # np.zeros((LengthOptim, LengthOptim)),
    # np.eye(LengthOptim),
    # np.zeros((LengthOptim, 1)),
    # np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
] # Energy from grid + energy from PV = DemandRequired
Aeq3 = np.c_[
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    # np.zeros((LengthOptim, LengthOptim)),
    # np.zeros((LengthOptim, LengthOptim)),
    # np.zeros((LengthOptim, 1)),
    # np.zeros((LengthOptim, LengthOptim)),
    alphaThermal2 * np.eye(LengthOptim),
    alphaThermal2 * np.eye(LengthOptim),
    -(1 - alphaThermal1)
    * (np.tri(LengthOptim, LengthOptim, -1) - np.tri(LengthOptim, LengthOptim, -2))
    + np.eye(LengthOptim),
    -np.eye(LengthOptim) * (Temperature_Ext)+ np.eye(LengthOptim)*np.r_[np.array([TemperatureInit]), np.zeros(LengthOptim - 1)]
] #thermal physical law

# zeros(LengthOptim)                      zeros(LengthOptim)                  zeros(LengthOptim)       zeros(LengthOptim,1)           zeros(LengthOptim)            -alphaThermal2*eye(LengthOptim)  diag(-(1-alphaThermal1)*ones(LengthOptim-1,1),-1)+eye(LengthOptim)

# Aeq3=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),gamma*np.eye(LengthOptim),-alpha*np.eye((LengthOptim),k=-1)+np.eye(LengthOptim),],axis=1)
# Aeq4=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.eye(LengthOptim)],axis=1)
# Aeq = np.concatenate([Aeq1, Aeq2, Aeq3], axis=0)
Aeq = np.concatenate([Aeq2, Aeq3], axis=0)
# print(Aeq)
# Beq1 = ProductionSeller
Beq2 = DemandRequired
Beq3 = (1 - 0) * np.r_[np.array([TemperatureInit]), np.zeros(LengthOptim - 1)]+0.1*IrradianceInertia/100
Beq = np.concatenate([ Beq2, Beq3], axis=0)
# Beq = np.concatenate([Beq1, Beq2, Beq3], axis=0)

A1 = np.c_[
    np.zeros((LengthOptim, LengthOptim)),
    np.eye((LengthOptim)),
    # np.zeros((1, LengthOptim)),
    # -1 * np.ones((1, LengthOptim)),
    # np.array([0]),
    # np.zeros((1, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.eye((LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim))
] #demand for supply from PV + demand for heating from PV < PV production
# A1 = np.concatenate([np.ones((LengthOptim)), np.zeros((LengthOptim)), -1*np.ones((LengthOptim)), 0,             np.zeros((LengthOptim)),  np.zeros((LengthOptim)), np.zeros((LengthOptim)),],axis=1)
#  ones(1,LengthOptim)    zeros(1,LengthOptim)        -1*ones(1,LengthOptim) 0                    zeros(1,LengthOptim)  zeros(1,LengthOptim)  zeros(1,LengthOptim)  ;  %sum demand <= sum need
# A2 = np.c_[
#     np.eye((LengthOptim)),
#     # -1 * np.eye((LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros(LengthOptim),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
#     np.zeros((LengthOptim, LengthOptim)),
# ]  # demand < production



# A = np.concatenate([A1, A2, A3, A4, A5, A6], axis=0)
A = np.concatenate([A1], axis=0)

b = np.r_[
    ProductionSeller
    # np.zeros((LengthOptim)),
    # np.zeros([LengthOptim]),
    # np.zeros([LengthOptim]),
    # np.array([0]),
    # np.array([0]),
]
lb = np.r_[
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
    # np.zeros((LengthOptim)),
    # np.array([0]),
    # np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    np.zeros((LengthOptim)),
    TemperatureMin,
    (alphaThermal1min) * np.ones(LengthOptim),
]
ub = np.r_[
    EmaxDaily * np.ones((LengthOptim)),
    EmaxDaily * np.ones((LengthOptim)),
    # EmaxDaily * np.ones((LengthOptim)),
    # EmaxDaily * np.ones((LengthOptim)),
    # np.array([EmaxDaily]),
    # EmaxDaily * np.ones((LengthOptim)),
    Pmaxchauf * np.ones((LengthOptim)),
    Pmaxchauf * np.ones((LengthOptim)),
    TemperatureMax,
    (alphaThermal1max) * np.ones(LengthOptim),
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
A = np.concatenate([A1, Aeq, np.eye(f.shape[0])], axis=0)
bl = np.concatenate([-9e99 * np.ones_like(b), Beq, lb], axis=0)
bu = np.concatenate([b, Beq, ub], axis=0)

constraints = LinearConstraint(A, bl, bu)
# integrality = Index_binary

res = milp(c=f, constraints=constraints)  # , integrality=integrality)
solution = res.x
print("Solution:", solution)
time = [0]
temperature_optim = [TemperatureInit]

# np.set_printoptions(linewidth=300)
output = ""
print("Output Temperature: ")
for i in range(LengthOptim - 1):
    output = output + str(solution[solution.shape[0] - 2 * LengthOptim + i]) + ","
    time.append(time[i] + 1)
    temperature_optim.append(solution[solution.shape[0] - 2 * LengthOptim + i + 1])
print(output)

PuissanceHeatPV = [0]
PuissanceHeatGrid = [0]
PuissanceHeattotal = [0]

for i in range(LengthOptim - 1):
    PuissanceHeatPV.append(solution[solution.shape[0] - 3 * LengthOptim + i])
    PuissanceHeatGrid.append(solution[solution.shape[0] - 4 * LengthOptim + i])
    PuissanceHeattotal.append(solution[solution.shape[0] - 4 * LengthOptim + i]+solution[solution.shape[0] - 3 * LengthOptim + i])

plt.figure()
plt.suptitle("Tint et Text")
plt.plot(
    time,
    temperature_optim,
    time,
    TemperatureMin,
    time,
    Temperature_Ext,
    time,
    TemperatureMax,
    time,
    Tamb,
    time,
    PuissanceHeattotal,
    time,
    500*solution[solution.shape[0]-1*LengthOptim:solution.shape[0]]
)
plt.legend(
    [
        "Temp optim",
        "Temperature Min",
        "Temperature Ext",
        "Temperature Max",
        "Temperature Amb",
        "Puissance for heat/cool (from PV & Grid)",
        "Alpha"
    ]
)
# plt.xticks(rotation=90)
plt.show()

# Puissance = [0]

# for i in range(LengthOptim - 1):
#     Puissance.append(solution[solution.shape[0] - 3 * LengthOptim + i])

plt.figure()
plt.suptitle("Puissance Consommée")
plt.plot(time, PuissanceHeatPV)
plt.plot(time, PuissanceHeatGrid)
plt.legend(["Puissance PV", "Puissance Grid"])
# plt.xticks(rotation=90)
plt.show()




x = np.linspace(0, LengthOptim,LengthOptim)#,LengthOptim) 
# Plot the array
plt.plot(x,Temperature_Ext, label='Temp Ext')
plt.plot(x,TemperatureMax, label='Temp Max')
plt.plot(x,solution[solution.shape[0]-2*LengthOptim:solution.shape[0]-1*LengthOptim], label='Temp Int')
plt.plot(x,1000*solution[solution.shape[0]-1*LengthOptim:solution.shape[0]], label='Alpha')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()

plt.show()





print("Required Energy: ")
output = ""
for i in range(LengthOptim - 1):
    output = output + str(solution[solution.shape[0] - 3 * LengthOptim + i]) + ","
print(output)
sys.stdout.flush()
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
