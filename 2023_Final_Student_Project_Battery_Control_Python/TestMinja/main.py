import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import datetime as dt
import math
from scipy.io import loadmat
from pandas import read_csv
import pandas as pd 
# import gurobipy as gp
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print(os.getcwd())

# mat1 = pd.read_csv('./2019.csv')
# mat2 = pd.read_csv('./output_2030.csv')

Timestep = 1/2 #time Step in hour
Window = 24 # Optimization for half a day
Length = int(Window/Timestep) #number of data points
NdataConsidered = 48*6 #computation for the whole week
Start = 0
IBCpercent = 0.51  # Initial Battery Capacity (IBC)of 80#MacBcap
BatteryCapacity = [80]#  Battery capacity
WHCRange = BatteryCapacity # Range of Battery capacity in Watt-Hour (WH)
MinCapPercent = 0


# # Price Scheme of Energy Buying from Grid at Agile Octopus Grid Buying Price (BP)
# load Buying_Selling_Price #Input of Daily Agile Octopus Buying Price (p/kWH)
# AgileOctopusSP=mat2["price"]
AgileOctopusSP = np.array([24.935777393676684, 22.86298909914984, 20.88294743405571, 18.83212878725165, 17.36113899, 17.289414265197433, 19.094966892660917, 20.486008821935897, 20.38369628098004, 18.82939277004111, 21.56618031582647, 24.780478866561488, 28.56068639623227, 49.67449394551728, 40.46525311, 42.50732278339712, 42.308176452095935, 50.29852804, 53.443038553435954, 58.32885509999403, 48.811405017438005, 41.05937135324314, 37.42461721, 38.74765518674431])
AgileOctopusSP=AgileOctopusSP[0:24].reshape(24,-1)
AgileOctopusBP = AgileOctopusSP*1.5 

# load PVPower_Demand # Input of computed wind power and demand
# Porigin=mat1["output"].iloc[0:24] # Final wind power in Wh at one minute basis
# Dorigin=mat1["SoloDemand"].iloc[0:24] # Final demand in Wh at one minute basis
# Porigin=mat2["PPprod"].iloc[0:24] # Final wind power in Wh at one minute basis
Porigin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.72807243930658, 1.147045601535538, 0.1816857567483509, 3.765877712588079, 2.5178341816685297, 3.2503037378265027, 6.657199260482738, 12.038989965583031, 13.24834220217527, 15.016140764385792, 14.554930900474856, 14.231453846323912, 13.662399160261192, 12.404847446319092, 10.401910207186845, 6.706052515])
# Dorigin=mat2["demand"].iloc[0:24] # Final demand in Wh at one minute basis
Dorigin = np.array([1.136140351, 1.183717949, 1.192298851, 1.193567251, 1.189019608, 12.30666667, 12.5830303, 17.69119048, 16.86888889, 20.31896296, 15.445, 4.556956522, 11.37988304, 3.249615385, 10.69116667, 1.552916667, 12.75207207, 12.44, 12.68304094, 12.24218391, 10.54, 8.950060606, 1.835444444, 1.189310345])
# # Strategy-1&2 to determine: the State of Charge (SoC) of the battery,
# # total energy sold to grid and total energy bought from grid

BPorigin = AgileOctopusBP # Hourly grid buying price for one year
SPorigin = AgileOctopusSP # Hourly grid selling price for one year 

Ndata =len(Porigin)
# t2=list(range(1,338))
#t2 = (datetime(2019,5,13,00,00,00):minutes(30):datetime(2019,5,19,23,59,00))'
# j = 0
P = Porigin
D = Dorigin

# for i in range(0,len(Porigin),30):
#      tmp = sum(Porigin[i:i+30])
#      P[j]= tmp[0]
#      tmp = sum(Dorigin[i:i+30])
#      D[j]= tmp[0]
#      sum(Dorigin[i:i+30])
#      j = j+1

BP=BPorigin
SP=SPorigin
# BP = interp1(t,BPorigin,t2)
# SP = interp1(t,SPorigin,t2)

# P = P[Start:Start+NdataConsidered]
# D= D[Start:Start+NdataConsidered]
BP = BP[Start:Start+NdataConsidered]
SP = SP[Start:Start+NdataConsidered]
time = list(range(1,25))

Today=dt.datetime(2023,1,2,00,00,00)

timeArray=[Today + dt.timedelta(minutes=60*x) for x in range(0, 24)]


DepreciationFactor =np.zeros(len(BatteryCapacity))
Batterycapacity =np.zeros(len(BatteryCapacity))
# ##optimization constraints
BPrice=BP[0:Length]
SPrice=SP[0:Length]
EfficiencyC = 0.9 # Battery Efficiency
EfficiencyD = 0.9
k = 1
Pmax = 6*max(max(P),max(D))
MaxBcap = BatteryCapacity[k-1]# Maximum battery capcity 
MinBcap = MinCapPercent*MaxBcap*0.2# Minimum battery capacity at 80# DoD
IBC = MaxBcap*IBCpercent # Initial Battery Capacity (IBC)of 60#MacBcap
Pbatmax = 1/5*MaxBcap*Timestep# max battery Power = 3.3kW# 3*max(P) #W  We consider there is no limit of battery power
SoCmax = BatteryCapacity[k-1] #kWh 

SoC =np.zeros(len(P))
Energysold =np.zeros(len(P))
Energybought =np.zeros(len(P))
FromGrid =np.zeros(len(P))
ToGrid =np.zeros(len(P))
Dvariable = 1000*np.ones(len(P))
SoCmax = BatteryCapacity[k-1] #kWh 
SoCmin = 0
SoCInit = IBC #kWh
Powerbat =np.zeros(len(P))
indice = 0

## start optimisation
for i in range(0,len(P),Length):# we iterate every 24 hours for the whole week
    indice = indice +1
    LengthOptim = min(i+Length,len(D))-i  # Because the size of P might not be a multiple of the Length used for the optimization
    Demand = D[i:i+LengthOptim]
    Production = P[i:i+LengthOptim]
    BPrice = BP[i:i+LengthOptim]
    SPrice = SP[i:i+LengthOptim] 
    f =np.r_[BPrice[:,0],  #imports from grid
             -SPrice[:,0],  #exports to grid
             np.zeros(LengthOptim), # Discharge power from battery
             np.zeros(LengthOptim), #charging power from battery
             np.zeros(LengthOptim), # auxiliary variable one (for import/export decoupling)
             np.zeros(LengthOptim), #auxiliary variable two (for import/export decoupling)
             np.zeros(LengthOptim), #binary variable one for EGridout,t is equal to 0 when EGridin,t ̸= 0,
             np.zeros(LengthOptim)] # binary variable two for EGridin,t is equal to 0 when EGridout,t ̸= 0,
    # f =np.r_([BPrice,-SPrice,np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1))])

    # Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))) #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144]
    Index_binary =np.r_[np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.ones(LengthOptim),np.ones(LengthOptim)]

    Aeq1=np.c_[np.eye(LengthOptim), np.zeros((LengthOptim,LengthOptim)), EfficiencyD*np.eye(LengthOptim), -np.eye(LengthOptim)/EfficiencyC, -np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)), np.zeros((LengthOptim,LengthOptim))]
    Aeq2=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim), -EfficiencyD*np.eye(LengthOptim) ,np.eye(LengthOptim)/EfficiencyC ,np.zeros((LengthOptim,LengthOptim)) ,-np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))]
    Aeq =np.concatenate([Aeq1,Aeq2],axis=0)
    Beq =np.concatenate([ Demand - Production , Production - Demand ],axis=0)
        
        
    A1=np.c_[np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim))]
    A2=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim))]
    A3=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,-Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim))]
    A4=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim))]
    A5=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-np.tril(np.ones((LengthOptim,LengthOptim))) ,np.tril(np.ones((LengthOptim,LengthOptim))) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim))]
    A6=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.tril(np.ones((LengthOptim,LengthOptim))) ,-np.tril(np.ones((LengthOptim,LengthOptim))) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim))]
    A7=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-Pbatmax*np.eye(LengthOptim)]
    A8=np.c_[np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,Pbatmax*np.eye(LengthOptim)]
    A =np.concatenate([A1,A2,A3,A4,A5,A6,A7,A8,Aeq, np.eye(f.shape[0])],axis=0)
    b = np.r_[np.zeros((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)), np.zeros((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)), (SoCmax-SoCInit)*np.ones((LengthOptim,1)), (SoCInit - SoCmin)*np.ones((LengthOptim,1)), np.zeros((LengthOptim,1)), Pbatmax*np.ones((LengthOptim,1))]
    lb =np.r_[np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)),  np.zeros((LengthOptim,1)),  np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1))]
    ub = np.r_[Pmax*np.ones((LengthOptim,1)),700*np.ones((LengthOptim,1)), Pbatmax*np.ones((LengthOptim,1)),  Pbatmax*np.ones((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)),  Pmax*np.ones((LengthOptim,1)),  np.ones((LengthOptim,1)),  np.ones((LengthOptim,1))]
    bu = np.concatenate([b[:,0],Beq, ub[:,0]],axis =0)
    bl = np.concatenate([-9e99*np.ones_like(b[:,0]),Beq, lb[:,0]],axis =0)

    constraints = LinearConstraint(A, bl, bu)
    integrality = Index_binary

    res = milp(c=f, constraints=constraints, integrality=integrality)
    solution=res.x



    SoC[i:i+LengthOptim]=-np.tril(np.ones((LengthOptim,LengthOptim)))@solution[LengthOptim*2:LengthOptim*3]+np.tril(np.ones((LengthOptim,LengthOptim)))@solution[LengthOptim*3:LengthOptim*4]+SoCInit*np.ones((LengthOptim,1))[:,0]
    PgridIN=solution[0:LengthOptim]
    PgridOUT=solution[LengthOptim:LengthOptim*2]
    Pbat=solution[LengthOptim*2:LengthOptim*3]-solution[LengthOptim*3:LengthOptim*4]
    solution1 = solution[LengthOptim*4+1:LengthOptim*5]
    solution2 = solution[LengthOptim*5+1:LengthOptim*6]
    Alpha = solution[LengthOptim*6+1:LengthOptim*7]
    Beta = solution[LengthOptim*7+1:LengthOptim*8]
    SoCInit = SoC[i+LengthOptim-1]
    
    for j in range(0,LengthOptim-1):
        tmp = PgridIN[j]#*BPrice[j]
        Energybought[i+j] = PgridIN[j]
        tmp = PgridOUT[j]#*SPrice[j]
        Energysold[i+j] =PgridOUT[j]
        FromGrid[i+j] = max(0,(D[i+j]-P[i+j]-max(0,Pbat[j])*EfficiencyD+max(0,-Pbat[j]/EfficiencyC)))
        ToGrid[i+j] = max(0, (P[i+j]-D[i+j]+max(0,Pbat[j]*EfficiencyD)-max(0,-Pbat[j]/EfficiencyC)))
        Powerbat [i:i+LengthOptim] = Pbat



# Create subplots with 3 rows and 1 column
fig = make_subplots(rows=4, cols=1, subplot_titles=("Power", "Price", "Energy", "SoC"))

# Plot 1: Power
fig.add_trace(go.Scatter(x=timeArray, y=P, mode='lines', name='Production'), row=1, col=1)
fig.add_trace(go.Scatter(x=timeArray, y=D, mode='lines', name='Demand'), row=1, col=1)
fig.add_trace(go.Scatter(x=timeArray, y=Powerbat, mode='lines', name='Battery'), row=1, col=1)

# Plot 2: Price
fig.add_trace(go.Scatter(x=timeArray, y=BP.flatten(), mode='lines', name='Buying Price'), row=2, col=1)
fig.add_trace(go.Scatter(x=timeArray, y=SP.flatten(), mode='lines', name='Selling Price'), row=2, col=1)

# Plot 3: Energy
fig.add_trace(go.Scatter(x=timeArray, y=Energybought, mode='lines', name='GridIN'), row=3, col=1)
fig.add_trace(go.Scatter(x=timeArray, y=Energysold, mode='lines', name='GridOUT'), row=3, col=1)
fig.add_trace(go.Scatter(x=timeArray, y=ToGrid, mode='lines', name='ToGrid'), row=3, col=1)
# Plot 4: SoC
fig.add_trace(go.Scatter(x=timeArray, y=SoC.flatten(), mode='lines', name='State of Charge'), row=4, col=1)

# Update layout with title and legends
# Update layout for full-width plotting area
fig.update_layout(
    height=1100,
    width=1600,               # Set width to take the full screen
    title_text="Power, Price, and Energy Graphs",
    showlegend=True,
    margin=dict(l=10, r=10, t=30, b=10)  # Reduce margins for a wider plot area
)
# Show the figure
fig.show()

