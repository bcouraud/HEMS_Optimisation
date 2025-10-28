import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import datetime as dt
import math
from scipy.io import loadmat
from pandas import read_csv
import pandas as pd 
import gurobipy as gp

mat1 = loadmat('PVPower_Demand.mat') #open a .mat file and converts it into arrays
mat2 = loadmat('Buying_Selling_Price.mat')
data = pd.read_csv("Finaldemandtime.csv") #python does not read datetime variables from .mat files we have to input a .csv

Timestep = 1/2 #time Step in hour
Window = 24 # Optimization for half a day
Length = int(Window/Timestep) #number of data points
NdataConsidered = 48*6; #computation for the whole week
Start = 0;
IBCpercent = 0.51;  # Initial Battery Capacity (IBC)of 80#MacBcap
BatteryCapacity = [15000];#  Battery capacity
WHCRange = BatteryCapacity; # Range of Battery capacity in Watt-Hour (WH)
MinCapPercent = 0;
# ##   Price computation
# #Price Scheme of Energy Buying from Grid at Grid Buying Price (BP)
HHBP = 20/100000;#20/100000; # High Hour Buying Price(HHBP),20Pence/kWh or O.2Pounds/kWh or 0.0002Pounds/Wh
LHBP = 11/100000;#11/100000; # Low Hour Buying Price (LHBP, 11Pence/kWh or 0.11Pounds/kWh or 0.00011Pounds/Wh 
#base1 = dt.datetime(2019,5,13)
numminute1=1440
Timeframe=list(range(1,1441))
##Timeframe = [base1 + dt.timedelta(minutes=x) for x in range(numminute1)] #creates an array with every minute of one day
#Timeframe = (dt.datetime(2019,5,13):minutes(1):dt.datetime(2019,5,14))';# Timeframe.Format = 'dd-MM-yyyy HH:mm:ss';#T1-3 duration will change based on low & high hour pricing
HHP1 = 7.5;
HHP2 = 22.5;
T1=Timeframe[:math.ceil(HHP1*60)+1]# python begins indexing at 0 and not 1 # Hours of  LHBP tariff 
T2=Timeframe[math.ceil(HHP1*60)+1:math.ceil(HHP2*60)+1];# Hours of HHBP tariff
T3 = Timeframe[math.ceil(HHP2*60)+1:1440];# Hours of LHBP tariff
LHBP1 = LHBP*np.ones((len(T1),1));# Low buying price for all the hours
HHBP1 = HHBP*np.ones((len(T2),1));# High buying price for all the hours
LHBP2 = LHBP*np.ones((len(T3),1));# Low buying price for all the hours
Daybuyprice1440Hrs = [*LHBP1,*HHBP1,*LHBP2];
# # Price Scheme of Energy Selling to Grid at Grid Selling Price (SP)
HHSP = 5.5/100000;#2/100000; # High Hour Selling Price(HHSP),2Pence/kWh or O.02Pounds/kWh or 0.00002Pounds/Wh
LHSP = 5.4999/100000;#1.99/100000; # Low Hour Selling Price (LHSP),1.99Pence/kWh or O.0199Pounds/kWh or 0.0000199Pounds/Wh
LHSP1 = LHSP*np.ones((len(T1),1));# Low selling price for all the hours
HHSP1 = HHSP*np.ones((len(T2),1));# High selling price for all the hours
LHSP2 = LHSP*np.ones((len(T3),1));# Low selling price for all the hours
Daysellprice48Hrs = [*LHSP1,*HHSP1,*LHSP2];
#base2 = dt.datetime(2019,5,13,00,00,00)
numminute2=10080
t=list(range(1,10081))
##t = [base2 + dt.timedelta(minutes=x) for x in range(numminute2)]
# t = (datetime(2019,5,13,00,00,00):minutes(1):datetime(2019,5,19,23,59,00))';
numminute3=337
ttemp=list(range(1,337))
##ttemp = [base2 + dt.timedelta(minutes=x*30) for x in range(numminute3)]
# ttemp = (datetime(2019,5,13,00,00,00):minutes(Timestep*60):datetime(2019,5,20,00,00,00))';

# # Price Scheme of Energy Buying from Grid at Agile Octopus Grid Buying Price (BP)
# load Buying_Selling_Price; #Input of Daily Agile Octopus Buying Price (p/kWH)
AgileOctopusBP=mat2["AgileOctopusBP"]
AgileOctopusSP=((AgileOctopusBP[0:len(ttemp)+1]))/100000/4
AgileOctopusBP=((AgileOctopusBP[0:len(ttemp)+1]))/100000

#Gridselling7days = interp1(ttemp, AgileOctopusBP(1:size(ttemp,1))/100000/4,t);%Outgoing Agile Octopus Selling Price(£/Wh)
Gridselling7Days =AgileOctopusSP[:338];#Outgoing Agile Octopus Selling Price(£/Wh)
#Gridbuying7Days = interp1(ttemp, AgileOctopusBP(1:size(ttemp,1))/100000,t);#Outgoing Agile Octopus Selling Price(£/Wh)
Gridbuying7Days =AgileOctopusBP[:338]


# ## Production and Consumption Data
# # Final Wind Power and Demand at Half Hourly Basis


# load PVPower_Demand; # Input of computed wind power and demand
Porigin=mat1["SoloPVPower"] # Final wind power in Wh at one minute basis
Dorigin=mat1["SoloDemand"] # Final demand in Wh at one minute basis
# # Strategy-1&2 to determine: the State of Charge (SoC) of the battery,
# # total energy sold to grid and total energy bought from grid

BPorigin = Gridbuying7Days # Hourly grid buying price for one year
SPorigin = Gridselling7Days # Hourly grid selling price for one year 

Ndata =len(Porigin)
t2=list(range(1,338))
#t2 = (datetime(2019,5,13,00,00,00):minutes(30):datetime(2019,5,19,23,59,00))';
j = 0;
P =np.zeros(len(t2));
D =np.zeros(len(t2));

for i in range(0,len(Porigin),30):
     P[j]= sum(Porigin[i:i+30])
     D[j]= sum(Dorigin[i:i+30])
     j = j+1;

BP=BPorigin[:337]#on trouve pas la premiere valeur
SP=SPorigin[:337]
# BP = interp1(t,BPorigin,t2);
# SP = interp1(t,SPorigin,t2);

P = P[Start:Start+NdataConsidered];
D= D[Start:Start+NdataConsidered];
BP = BP[Start:Start+NdataConsidered];
SP = SP[Start:Start+NdataConsidered];
time = t2[Start:Start+NdataConsidered];

Today=dt.datetime(2019,5,13,00,00,00)

timeenjour=[Today + dt.timedelta(minutes=30*x) for x in range(0, 288)]


DepreciationFactor =np.zeros(len(BatteryCapacity));
Batterycapacity =np.zeros(len(BatteryCapacity));
# ##optimization constraints
BPrice=BP[0:Length]
SPrice=SP[0:Length]
EfficiencyC = 0.87; # Battery Efficiency
EfficiencyD = 0.87;
k = 1;    
Pmax = 6*max(max(P),max(D));
MaxBcap = BatteryCapacity[k-1];# Maximum battery capcity in WHr [Variable Parameter]
MinBcap = MinCapPercent*MaxBcap*0.2;# Minimum battery capacity at 80# DoD
IBC = MaxBcap*IBCpercent; # Initial Battery Capacity (IBC)of 60#MacBcap
Pbatmax = 2*MaxBcap*Timestep;# max battery Power = 3.3kW# 3*max(P); #W  We consider there is no limit of battery power
SoCmax = BatteryCapacity[k-1]; #kWh 
M = 50*max(SoCmax,Pbatmax);
SoC =np.zeros(len(P));
Energysold =np.zeros(len(P));
Energybought =np.zeros(len(P));
FromGrid =np.zeros(len(P));
ToGrid =np.zeros(len(P));
Dvariable = 1000*np.ones(len(P));
SoCmax = BatteryCapacity[k-1]; #kWh 
SoCmin = 0;
SoCInit = IBC; #kWh
Powerbat =np.zeros(len(P));
Pchauff=np.zeros(len(P))
indice = 0;

## Nouvelle variables ##
deltaT=10 ## un delta T choisi arbitrairement
K=0.9 ## un Coeff K qui depend de l'inertie du batiment
G=0.6 ## un Coeff G qui depend du batiment
V=1000 ## le volume du batiment

Consigne=np.concatenate([(21)*np.ones((72,1)),(21)*np.ones((144,1)),(21)*np.ones((72,1))],axis=0)

Text=np.concatenate([5*np.ones((144)),14*np.ones((144))],axis=0) ## Temperature sur une semaine 288 valeur par pas de temps 30 min


## start optimisation
for i in range(0,len(P),Length):# we iterate every 24 hours for th whole week
    indice = indice +1;
    LengthOptim = min(i+Length,len(D))-i;  # Because the size of P might not be a multiple of the Length used for the optimization
    Demand = D[i:i+LengthOptim];
    Production = P[i:i+LengthOptim];
    BPrice = BP[i:i+LengthOptim];
    SPrice = SP[i:i+LengthOptim]; 
    
    f =np.concatenate([BPrice[:,0],-SPrice[:,0],np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim)]);

    Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
    
    Aeq1=np.concatenate([np.eye(LengthOptim), np.zeros((LengthOptim,LengthOptim)), EfficiencyD*np.eye(LengthOptim), -np.eye(LengthOptim)/EfficiencyC, -np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)), np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    Aeq2=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim), -EfficiencyD*np.eye(LengthOptim) ,np.eye(LengthOptim)/EfficiencyC ,np.zeros((LengthOptim,LengthOptim)) ,-np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    Aeq3=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.eye(LengthOptim)],axis=1)
    Aeq =np.concatenate([Aeq1,Aeq2,Aeq3],axis=0)
            
        
    A1=np.concatenate([np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A2=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A3=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,-Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A4=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A5=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-np.tril(np.ones((LengthOptim,LengthOptim))) ,np.tril(np.ones((LengthOptim,LengthOptim))) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A6=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.tril(np.ones((LengthOptim,LengthOptim))) ,-np.tril(np.ones((LengthOptim,LengthOptim))) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A7=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-Pbatmax*np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A8=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,Pbatmax*np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A9=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.eye(LengthOptim)],axis=1)
    A =np.concatenate([A1,A2,A3,A4,A5,A6,A7,A8,A9],axis=0)
        
    Beq1=Demand - Production
    Beq2=Production-Demand
    Beq3=G*V*(Consigne[i]-Text[i])*np.ones((LengthOptim))
    Beq =np.concatenate([Beq1,Beq2,Beq3],axis=0);
    
    b = np.concatenate([np.zeros((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)), np.zeros((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)), (SoCmax-SoCInit)*np.ones((LengthOptim,1)), (SoCInit - SoCmin)*np.ones((LengthOptim,1)), np.zeros((LengthOptim,1)), Pbatmax*np.ones((LengthOptim,1)),Pmax*np.ones((LengthOptim,1))])
    lb =np.concatenate([np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)),  np.zeros((LengthOptim,1)),  np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1))])
    ub = np.concatenate([Pmax*np.ones((LengthOptim,1)),Pmax*np.ones((LengthOptim,1)), Pbatmax*np.ones((LengthOptim,1)),  Pbatmax*np.ones((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)),  Pmax*np.ones((LengthOptim,1)),  np.ones((LengthOptim,1)),  np.ones((LengthOptim,1)),Pmax*np.ones((LengthOptim,1))])
    b=b[:,0]

#     #if we wanted to change the battery power for discharge for the night (to
#     #follow a simulation where price was very high for selling and see if the simulation is able to find a good behaviour)
#     #             ub(LengthOptim*2+21:LengthOptim*2+24)=3;
#     # ub(LengthOptim*3+21:LengthOptim*3+24)=3;
#     #             x0=[];
#     #              options=optimoptions('intlinprog','Display','off');    bounds=(lbbis[indice-1],ubbis[indice-1])
#     #             options.Display = 'off';

    model=gp.Model()
    x=model.addMVar(432,ub=ub,lb=lb)
    x.obj=f
    model.addConstr(A@x<=b)
    model.addConstr(Aeq@x==Beq)
    model.optimize()
    solution=x.X
 
    SoC[i:i+LengthOptim]=-np.tril(np.ones((LengthOptim,LengthOptim)))@solution[LengthOptim*2:LengthOptim*3]+np.tril(np.ones((LengthOptim,LengthOptim)))@solution[LengthOptim*3:LengthOptim*4]+SoCInit*np.ones((LengthOptim,1))[:,0]
    PgridIN=solution[0:LengthOptim]
    PgridOUT=solution[LengthOptim:LengthOptim*2]
    Pbat=solution[LengthOptim*2:LengthOptim*3]-solution[LengthOptim*3:LengthOptim*4]
    solution1 = solution[LengthOptim*4:LengthOptim*5]
    solution2 = solution[LengthOptim*5:LengthOptim*6]
    Alpha = solution[LengthOptim*6:LengthOptim*7]
    Beta = solution[LengthOptim*7:LengthOptim*8]
    Pchauffage=solution[LengthOptim*8:LengthOptim*9]
    SoCInit = SoC[i+LengthOptim-1];
    
    for j in range(0,LengthOptim):
        Energybought[i+j-1] = PgridIN[j]*BPrice[j]
        Energysold[i+j-1] =PgridOUT[j]*SPrice[j]
        FromGrid[i+j-1] = max(0,(D[i+j-1]-P[i+j-1]-max(0,Pbat[j])*EfficiencyD+max(0,-Pbat[j]/EfficiencyC)))
        ToGrid[i+j-1] = max(0, (P[i+j-1]-D[i+j-1]+max(0,Pbat[j]*EfficiencyD)-max(0,-Pbat[j]/EfficiencyC)))
        Powerbat [i:i+LengthOptim] = Pbat;
        Pchauff[i:i+LengthOptim]=Pchauffage

plt.figure(1)
plt.suptitle('Puissances')
plt.plot(timeenjour,P, timeenjour, D, timeenjour, Powerbat)
plt.legend(['Production', 'Demand','Battery'])
plt.xticks(rotation=90)

plt.figure(2)
plt.suptitle('Prix')
plt.plot(timeenjour,BP, timeenjour, SP)
plt.legend(['Buying Price', 'Selling Price'])
plt.xticks(rotation=90)
    
plt.figure(3)
plt.suptitle('Puissance de chauffage')
plt.plot(timeenjour,Pchauff)
plt.legend(['Pchauffage'])
plt.xticks(rotation=90)

plt.figure(4)
plt.suptitle('Tint et Text')
plt.plot(timeenjour,Text,timeenjour,Consigne)
plt.legend(['Temp ext','Temp int'])
plt.xticks(rotation=90)
