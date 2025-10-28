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


weightEnv = 0.5
weightCost = 0.5


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
        11000.4,
        10600.5,
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

BPrice = weightEnv * CO2emissions + weightCost * GridPrice
EmaxDaily = 100000


ProductionSeller = EmaxDaily * np.ones(LengthOptim)
ProductionSeller = np.array(
    [0, 0, 0, 0,0,0,0, 1, 2, 2,2,2,2,3, 3, 2, 2,2,2,1,1,0, 0, 0]
).reshape(LengthOptim,1)

DemandRequired = 0*np.ones(LengthOptim)


E_cycle = [5, 4, 2]

f = np.r_[
    BPrice, #Demand supplied from the grid/gas engine
    0*BPrice, #Demand supplied from PV
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
]
#pour un appareil dont le temps de travail est de 3h, on crée 3 variables entières 
# (qui seront les booléens, comme tu disais) de 22 éléments (pour pas surcharger, cf plus bas). 
# Une pour le temps auquel l'appareil démarre (donc ça peut pas démarrer à 23h sinon ça finit pas,
#  d'où le 22 éléments), une  pour le temps auquel l'appareil fait son deuxième time step de travail
#  et une pour dire à quel moment il fait son dernier timestep

#  Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
Index_binary =np.array(list(range(len(f)-3*LengthOptim+2+1+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
# Define the cutoff index
cutoff_index = len(f) - LengthOptim * 3 + 3

# Create the array with 0s up to the cutoff index and 1s after
integrality = np.zeros(len(f))
integrality[cutoff_index:] = 1

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


Aeq1 = np.c_[ #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))

    np.zeros((LengthOptim, LengthOptim)),
    np.zeros((LengthOptim, LengthOptim)),
    First_Slot_Appliance, #appliance
    -1/2*Second_Slot_Appliance,#appliance
    -1/2*Third_Slot_Appliance,#appliance
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
#Une avec 24 lignes qui dit (comme tu as proposé) 
# que la variable 1 à t = 1/2*("variable 2 à t+1" + "variable 3 à t+2")
Beq1 = np.zeros((LengthOptim,1))

Aeq2 = np.c_[ #to ensure the appliance will run and only once
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.ones((1, LengthOptim-2)),#appliance
    np.zeros((1, LengthOptim-1)),#appliance
    np.zeros((1, LengthOptim)),#appliance
] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
Beq2 = [[1]]
Aeq3 = np.c_[ #to ensure the appliance will run and only once
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim-2)),#appliance
    np.ones((1, LengthOptim-1)),#appliance
    np.zeros((1, LengthOptim)),#appliance
] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
Beq3 = [[1]]
Aeq4 = np.c_[ #to ensure the appliance will run and only once
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim)),
    np.zeros((1, LengthOptim-2)),#appliance
    np.zeros((1, LengthOptim-1)),#appliance
    np.ones((1, LengthOptim)),#appliance
] #une seule ligne pour dire que la somme des éléments de chacune de ces variables vaut 1
Beq4 =[[1]]


Aeq5 = np.c_[ #Demand = demand from the cycles
    np.eye(LengthOptim, LengthOptim),
    np.eye(LengthOptim, LengthOptim),
    -E_cycle[0]*np.eye(LengthOptim, LengthOptim-2), #appliance
    -E_cycle[1]*np.eye(LengthOptim, LengthOptim-1),#appliance
    -E_cycle[2]*np.eye(LengthOptim, LengthOptim),#appliance
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
Beq5 = np.zeros((LengthOptim,1))


A1 = np.c_[ #Demand = demand from the cycles
    np.zeros((LengthOptim, LengthOptim)),
    np.eye(LengthOptim, LengthOptim),
    np.zeros((LengthOptim, LengthOptim-2)), #appliance
    np.zeros((LengthOptim, LengthOptim-1)),#appliance
    np.zeros((LengthOptim, LengthOptim)),#appliance
] #to ensure the appliance will run its real cycle (= 3 consecutive slots for example here))
B1 = ProductionSeller

lb = np.r_[
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

]
ub = np.r_[
    EmaxDaily * np.ones((LengthOptim)),
    max(ProductionSeller) * np.ones((LengthOptim)),
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
A = np.concatenate([A1, Aeq1, Aeq2, Aeq3, Aeq4, Aeq5, np.eye(f.shape[0])], axis=0)
bl = np.concatenate([np.zeros((LengthOptim,1)),Beq1, Beq2, Beq3, Beq4, Beq5, lb.reshape((lb.shape[0], 1))], axis=0)
bu = np.concatenate([B1, Beq1, Beq2, Beq3, Beq4, Beq5, ub.reshape((ub.shape[0], 1))], axis=0)

constraints = LinearConstraint(A, bl.flatten(), bu.flatten())

res = milp(c=f, constraints=constraints, integrality=integrality)
solution = res.x
print("Solution:", solution)

DemandfromGrid= solution[0:LengthOptim]
DemandfromPV= solution[LengthOptim:LengthOptim*2]
Appliance_start = solution[f.shape[0]-LengthOptim*3+3:f.shape[0]-LengthOptim*2+1]
Appliance_cycle2 = solution[f.shape[0]-LengthOptim*2+1:f.shape[0]-LengthOptim]
Appliance_cycle3 = solution[f.shape[0]-LengthOptim:f.shape[0]+1]

# Save each matrix to a separate text file
Aeq = np.concatenate([Aeq1, Aeq2, Aeq3, Aeq4, Aeq5], axis=0)
beq = np.concatenate([Beq1, Beq2, Beq3, Beq4, Beq5], axis=0)
Amax = np.concatenate([A1,np.eye(f.shape[0])], axis=0)
Amin = np.concatenate([np.eye(f.shape[0])], axis=0)
bmax = np.concatenate([B1, ub.reshape((ub.shape[0], 1))], axis=0)
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

for i in range(LengthOptim - 1):
    output = output + str(solution[solution.shape[0] - 2 * LengthOptim + i]) + ","
    time.append(time[i] + 1)
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
