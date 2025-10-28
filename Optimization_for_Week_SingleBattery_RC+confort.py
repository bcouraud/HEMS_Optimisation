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
Tin=np.zeros(len(P))
indice = 0;

Treel=[16.200958,16.440956,16.520958,16.550957,16.510956,15.940957,15.630957	,15.330957,	15.190957,	15.710958,	15.400957	,14.950957	,15.380957	,15.410957	,16.240957,	15.540957,	14.940957	,14.940957,	14.830957,	12.520957,	10.890957,	9.400957,	8.830957,	8.590958,	10.150957,	10.100957,	10.090957,	10.230957,	10.200957	,10.530957,	9.660957,	11.110957,	14.610957	,16.560957	,17.840958,	18.410957,	19.070957,	19.140957,	18.990957,	18.100958,	15.600957,	12.790957,	11.620957,	10.830957,	10.260957,	9.900957,	11.2509575	,10.550957,	10.320957,	11.540957,	12.260957,	12.950957	,13.210958	,13.050957,	12.570957	,13.240957	,15.260957	,13.550957	,13.970957,	15.130957	,15.200957	,15.180957	,14.700957	,14.890957,	14.060957,	13.700957,	13.330957,	13.420958,	12.760957,	10.150957,	9.360957	,8.110957	,7.090957	,6.390957,	6.040957,	5.6809573,	5.370957,	5.140957	,4.880957,	6.420957,	10.060957,	13.680957,	14.790957,	14.730957,	14.320957,	14.240957	,14.030957,	14.060957,12.670958,	12.380957,	11.240957,	9.200957,	8.5409565,	7.630957,7.270957,	7.340957,	7.090957	,7.030957	,6.880957,	7.020957,	7.130957,	7.140957	,6.610957,	8.070957	,11.260957	,14.390957,	15.270957,	15.920958,	16.570957,	16.590958,	16.470957,	15.790957,	13.930957,	12.570957	,8.800957	,6.9709573,	6.580957,	6.440957	,6.370957,	6.170957,	5.730957,	5.580957,	5.6809573,	5.870957,	5.840957	,5.7609572	,5.9109573,	8.360957,	12.900957	,15.410957,	17.190958	,17.750957	,17.540956	,18.000957	,18.080957	,16.680958,	12.900957,	10.810957,	10.130957	,9.260957,	9.200957,	8.840958,	8.240957,	7.960957]

#0h-7h 0°C --- 7h-19h 20°C --- 19h-23h 0°C

JourOffHaute=np.concatenate([15*np.ones((2,1)),15*np.ones((44,1)),24*np.ones((2,1))],axis=0)
JourOffBasse=np.concatenate([5*np.ones((10,1)),5*np.ones((28,1)),5*np.ones((10,1))],axis=0)

JourTravailHaute=np.concatenate([18*np.ones((2,1)),18*np.ones((12,1)),24*np.ones((26,1)),20*np.ones((8,1))],axis=0)
JourTravailBasse=np.concatenate([5*np.ones((2,1)),5*np.ones((12,1)),20*np.ones((26,1)),5*np.ones((8,1))],axis=0)

Lundi=JourTravailBasse
Mardi=JourTravailBasse
Mercredi=JourTravailBasse
Jeudi=JourTravailBasse
Vendredi=JourTravailBasse
Samedi=JourOffBasse
ConsigneB=np.concatenate([Lundi,Mardi,Mercredi,Jeudi,Vendredi,Samedi],axis=0)

LundiH=JourTravailHaute
MardiH=JourTravailHaute
MercrediH=JourTravailHaute
JeudiH=JourTravailHaute
VendrediH=JourTravailHaute
SamediH=JourOffHaute
ConsigneH=np.concatenate([LundiH,MardiH,MercrediH,JeudiH,VendrediH,SamediH],axis=0)

T=np.concatenate([Treel,Treel],axis=0)
T1=np.concatenate([Treel,Treel,Treel],axis=0)

Tint0=Treel[0];

ModeConfort = input("Entrez le niveau de confort entre 1 et 8  (plus c'est haut plus c'est confort) :")
coeffconfort=int(ModeConfort)*0.00001

## start optimisation
for i in range(0,len(P),Length):# we iterate every 24 hours for th whole week
    indice = indice +1;
    LengthOptim = min(i+Length,len(D))-i;  # Because the size of P might not be a multiple of the Length used for the optimization
    Demand = D[i:i+LengthOptim];
    Production = P[i:i+LengthOptim];
    BPrice = BP[i:i+LengthOptim];
    SPrice = SP[i:i+LengthOptim]; 
    ConsigneHaute=ConsigneH[i:i+LengthOptim]
    ConsigneBasse=ConsigneB[i:i+LengthOptim]
    Text=T1[i+1:i+LengthOptim+1]
 
    pas=1/2
    C=2000000#100
    R=1/1.3#20
    beq3Mat=Text*(pas/C*R)*np.ones(LengthOptim)
    beq3Mat[0]=Tint0

    f =np.concatenate([BPrice[:,0],-SPrice[:,0],np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),np.zeros(LengthOptim),-coeffconfort*np.ones((LengthOptim))]);#coeff(consigne-tint)

    Index_binary =np.array(list(range(len(f)-2*LengthOptim+1,len(f)+1))); #[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
    
    Aeq1=np.concatenate([np.eye(LengthOptim), np.zeros((LengthOptim,LengthOptim)), EfficiencyD*np.eye(LengthOptim), -np.eye(LengthOptim)/EfficiencyC, -np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)), np.zeros((LengthOptim,LengthOptim)),-1*np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),],axis=1)
    Aeq2=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim), -EfficiencyD*np.eye(LengthOptim) ,np.eye(LengthOptim)/EfficiencyC ,np.zeros((LengthOptim,LengthOptim)) ,-np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),1*np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),],axis=1)
    Aeq3=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),-(pas/C*R)*np.eye((LengthOptim),k=-1),-np.eye((LengthOptim),k=-1)+(pas/C*R)*np.eye((LengthOptim),k=-1)+np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),],axis=1)
    Aeq4=np.concatenate([np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),-np.eye(LengthOptim),np.eye(LengthOptim)],axis=1)
    Aeq =np.concatenate([Aeq1,Aeq2,Aeq3,Aeq4],axis=0)
            
        
    A1=np.concatenate([np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A2=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A3=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,-Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A4=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,Pmax*np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A5=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-np.tril(np.ones((LengthOptim,LengthOptim))) ,np.tril(np.ones((LengthOptim,LengthOptim))) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A6=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.tril(np.ones((LengthOptim,LengthOptim))) ,-np.tril(np.ones((LengthOptim,LengthOptim))) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A7=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,-Pbatmax*np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A8=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.eye(LengthOptim) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,Pbatmax*np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A9=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.eye(LengthOptim),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A10=np.concatenate([np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)) ,np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim)),np.zeros((LengthOptim,LengthOptim))],axis=1)
    A =np.concatenate([A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A10],axis=0)
        
    Beq1=Demand- Production
    Beq2=Production-Demand
    Beq3=beq3Mat
    Beq4=ConsigneHaute*np.ones((LengthOptim,1))
    Beq =np.concatenate([Beq1,Beq2,Beq3,Beq4[:,0]],axis=0);

    b = np.concatenate([np.zeros((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)), np.zeros((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)), (SoCmax-SoCInit)*np.ones((LengthOptim,1)), (SoCInit - SoCmin)*np.ones((LengthOptim,1)), np.zeros((LengthOptim,1)), Pbatmax*np.ones((LengthOptim,1)),Pmax*np.ones((LengthOptim,1)),np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1))])
    lb =np.concatenate([np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)),  np.zeros((LengthOptim,1)),  np.zeros((LengthOptim,1)), np.zeros((LengthOptim,1)),np.zeros((LengthOptim,1)),ConsigneBasse,-100*np.ones((LengthOptim,1))])
    ub = np.concatenate([Pmax*np.ones((LengthOptim,1)),Pmax*np.ones((LengthOptim,1)), Pbatmax*np.ones((LengthOptim,1)),  Pbatmax*np.ones((LengthOptim,1)), Pmax*np.ones((LengthOptim,1)),  Pmax*np.ones((LengthOptim,1)),  np.ones((LengthOptim,1)),  np.ones((LengthOptim,1)),Pmax*np.ones((LengthOptim,1)),ConsigneHaute,100*np.ones((LengthOptim,1))])
    b=b[:,0]

    model=gp.Model()
    x=model.addMVar(528,ub=ub,lb=lb)
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
    Tint=solution[LengthOptim*9:LengthOptim*10]
    ConsigneHautee=solution[LengthOptim*10:LengthOptim*11]
    print('Temperature interieure :',Tint)
    Tint0=Tint[47]
    SoCInit = SoC[i+LengthOptim-1];
    
    for j in range(0,LengthOptim):
        Energybought[i+j-1] = PgridIN[j]*BPrice[j]
        Energysold[i+j-1] =PgridOUT[j]*SPrice[j]
        FromGrid[i+j-1] = max(0,(D[i+j-1]-P[i+j-1]-max(0,Pbat[j])*EfficiencyD+max(0,-Pbat[j]/EfficiencyC)))
        ToGrid[i+j-1] = max(0, (P[i+j-1]-D[i+j-1]+max(0,Pbat[j]*EfficiencyD)-max(0,-Pbat[j]/EfficiencyC)))
        Powerbat [i:i+LengthOptim] = Pbat;
        Pchauff[i:i+LengthOptim]=Pchauffage
        Tin[i:i+LengthOptim]=Tint


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
plt.plot(timeenjour,T,timeenjour,ConsigneH,timeenjour,ConsigneB,timeenjour,Tin,)
plt.legend(['Temp ext','Temp int max','Temp int min',"Temp int réelle"])
plt.xticks(rotation=90)

plt.figure(5)
plt.suptitle('Tint et Text Lundi')
plt.plot(timeenjour[0:47],T[0:47],timeenjour[0:47],ConsigneH[0:47],timeenjour[0:47],ConsigneB[0:47],timeenjour[0:47],Tin[0:47],)
plt.legend(['Temp ext','Temp int max','Temp int min',"Temp int réelle"])
plt.xticks(rotation=90)

plt.figure(6)
plt.suptitle('Tint et Text Samedi')
plt.plot(timeenjour[239:287],T[239:287],timeenjour[239:287],ConsigneH[239:287],timeenjour[239:287],ConsigneB[239:287],timeenjour[239:287],Tin[239:287],)
plt.legend(['Temp ext','Temp int max','Temp int min',"Temp int réelle"])
plt.xticks(rotation=90)

plt.figure(7)
plt.suptitle('TPuissance et prix Lundi')
plt.plot(timeenjour[0:47],Pchauff[0:47],timeenjour[0:47],30000*BP[0:47],)
plt.legend(['Pchauff',"Prix d'achat"])
plt.xticks(rotation=90)

plt.figure(8)
plt.suptitle('TPuissance et prix Samedi')
plt.plot(timeenjour[239:287],Pchauff[239:287],timeenjour[239:287],30000*BP[239:287],)
plt.legend(['Pchauff',"Prix d'achat"])
plt.xticks(rotation=90)
