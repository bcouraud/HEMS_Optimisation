close all
clear all
% Price Scheme of Energy Buying from Grid at Grid Buying Price (BP)
% High Hour Buying Price is assumed to be 20Pence or 0.2Pounds per kWH,
% which is 0.0002 Pounds per WH (Divided by 1000 to covert from Pounds/kWH
% to Pounds/WH, same applies to all other pricing schemes

Timestep = 1/2; %Time Step in hour
Window = 1*24;% Optimization for half a day
Length = Window/Timestep; %number of data points
NdataConsidered = 48*6; %computation for the whole week
Start = 0;
IBCpercent = 0.51;  % Initial Battery Capacity (IBC)of 80%MacBcap
BatteryCapacity = [15000];%  Battery capacity
WHCRange = BatteryCapacity'; % Range of Battery capacity in Watt-Hour (WH)
MinCapPercent = 0;

%%   Price computation
%Price Scheme of Energy Buying from Grid at Grid Buying Price (BP)
HHBP = 20/100000;%20/100000; % High Hour Buying Price(HHBP),20Pence/kWh or O.2Pounds/kWh or 0.0002Pounds/Wh
LHBP = 11/100000;%11/100000; % Low Hour Buying Price (LHBP, 11Pence/kWh or 0.11Pounds/kWh or 0.00011Pounds/Wh 
Timeframe = (datetime(2019,5,13):minutes(1):datetime(2019,5,14))';
Timeframe.Format = 'dd-MM-yyyy HH:mm:ss';%T1-3 duration will change based on low & high hour pricing
HHP1 = 7.5;
HHP2 = 22.5;
T1 = Timeframe(1:HHP1*60+1);% Hours of  LHBP tariff 
T2 = Timeframe(HHP1*60+2:HHP2*60+1);% Hours of HHBP tariff
T3 = Timeframe(HHP2*60+2:1440);% Hours of LHBP tariff
LHBP1 = LHBP*ones(length(T1),1);% Low buying price for all the hours
HHBP1 = HHBP*ones(length(T2),1);% High buying price for all the hours
LHBP2 = LHBP*ones(length(T3),1);% Low buying price for all the hours
Daybuyprice1440Hrs = [LHBP1;HHBP1;LHBP2];
% Price Scheme of Energy Selling to Grid at Grid Selling Price (SP)
HHSP = 5.5/100000;%2/100000; % High Hour Selling Price(HHSP),2Pence/kWh or O.02Pounds/kWh or 0.00002Pounds/Wh
LHSP = 5.4999/100000;%1.99/100000; % Low Hour Selling Price (LHSP),1.99Pence/kWh or O.0199Pounds/kWh or 0.0000199Pounds/Wh
LHSP1 = LHSP*ones(length(T1),1);% Low selling price for all the hours
HHSP1 = HHSP*ones(length(T2),1);% High selling price for all the hours
LHSP2 = LHSP*ones(length(T3),1);% Low selling price for all the hours
Daysellprice48Hrs = [LHSP1;HHSP1;LHSP2];
t = (datetime(2019,5,13,00,00,00):minutes(1):datetime(2019,5,19,23,59,00))';
ttemp = (datetime(2019,5,13,00,00,00):minutes(Timestep*60):datetime(2019,5,20,00,00,00))';

% Price Scheme of Energy Buying from Grid at Agile Octopus Grid Buying Price (BP)
load Buying_Selling_Price; %Input of Daily Agile Octopus Buying Price (p/kWH)
Gridselling7days = interp1(ttemp, AgileOctopusBP(1:size(ttemp,1))/100000/4,t);%Outgoing Agile Octopus Selling Price(£/Wh)
Gridbuying7Days = interp1(ttemp, AgileOctopusBP(1:size(ttemp,1))/100000,t);%Outgoing Agile Octopus Selling Price(£/Wh)

%% Production and Consumption Data
% Final Wind Power and Demand at Half Hourly Basis
load PVPower_Demand; % Input of computed wind power and demand
Porigin = SoloPVPower; % Final wind power in Wh at one minute basis
Dorigin = SoloDemand;% Final demand in Wh at one minute basis
% Strategy-1&2 to determine: the State of Charge (SoC) of the battery,
% total energy sold to grid and total energy bought from grid
BPorigin = Gridbuying7Days ;% Hourly grid buying price for one year
SPorigin = Gridselling7days ;% Hourly grid selling price for one year 

Ndata =  size(Porigin,1);
t2 = (datetime(2019,5,13,00,00,00):minutes(30):datetime(2019,5,19,23,59,00))';
j = 1;
P = zeros(size(t2));
D = zeros(size(t2));
for i =1:30:size(Porigin,1)
    P(j)= sum(Porigin(i:i+29));
    D(j)= sum(Dorigin(i:i+29));
    j = j+1;
end
BP = interp1(t,BPorigin,t2);
SP = interp1(t,SPorigin,t2);
P = P(Start+1:Start+NdataConsidered);
D= D(Start+1:Start+NdataConsidered);
BP = BP(Start+1:Start+NdataConsidered);
SP = SP(Start+1:Start+NdataConsidered);
time = t2(Start+1:Start+NdataConsidered);
DepreciationFactor = zeros(length(BatteryCapacity),1);
Batterycapacity = zeros(length(BatteryCapacity),1);
%%optimization constraints
BPrice = BP(1:Length); 
SPrice = SP(1:Length); 
EfficiencyC = 0.87; % Battery Efficiency
EfficiencyD = 0.87;
k = 1;    
Pmax = 6*max(max(P),max(D));
MaxBcap = BatteryCapacity(k);% Maximum battery capcity in WHr [Variable Parameter]
MinBcap = MinCapPercent*MaxBcap*0.2;% Minimum battery capacity at 80% DoD
IBC = MaxBcap*IBCpercent; % Initial Battery Capacity (IBC)of 60%MacBcap
Pbatmax = 2*MaxBcap*Timestep;% max battery Power = 3.3kW% 3*max(P); %W  We consider there is no limit of battery power
SoCmax = BatteryCapacity(k); %kWh 
M = 50*max(SoCmax,Pbatmax);
SoC = zeros(length(P),1);
Energysold = zeros(length(P),1);
Energybought = zeros(length(P),1);
FromGrid = zeros(length(P),1);
ToGrid = zeros(length(P),1);
Dvariable = 1000*ones(length(P),1);
SoCmax = BatteryCapacity(k); %kWh 
SoCmin = 0;
SoCInit = IBC; %kWh
Powerbat = zeros(length(P),1);
indice = 0;

%% start optimisation
for i = 1:Length:length(P) % we iterate every 24 hours for th whole week
    indice = indice +1;
    LengthOptim = min(i+Length,length(D))-i+1;  % Because the size of P might not be a multiple of the Length used for the optimization
    Demand = D(i:i+LengthOptim-1);
    Production = P(i:i+LengthOptim-1);
    BPrice = BP(i:i+LengthOptim-1);
    SPrice = SP(i:i+LengthOptim-1); 
    f = [BPrice;-SPrice;zeros(LengthOptim,1); zeros(LengthOptim,1);zeros(LengthOptim,1);zeros(LengthOptim,1);zeros(LengthOptim,1); zeros(LengthOptim,1) ];

    Index_binary = size(f,1)-2*LengthOptim+1:size(f,1); %[121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144];
    Aeq = [eye(LengthOptim) zeros(LengthOptim) EfficiencyD*eye(LengthOptim) -eye(LengthOptim)/EfficiencyC -eye(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim); %Pgrin in = Demand - Production + Pbat_recharge/efficiency -Pbat_Discharge*Efficiency +X1   || X1 is here to make t possible to have Pgrid IN !=0 when Pgrid out =0
           zeros(LengthOptim) eye(LengthOptim) -EfficiencyD*eye(LengthOptim) eye(LengthOptim)/EfficiencyC zeros(LengthOptim) -eye(LengthOptim) zeros(LengthOptim) zeros(LengthOptim)]; %Pgrin out = - Demand + Production - Pbat_recharge/efficiency + Pbat_Discharge*Efficiency +X2 || X2 is here to make t possible to have Pgrid OUT !=0 when Pgrid in =0
    beq = [ Demand - Production ; Production - Demand ];

    A = [eye(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) -Pmax*eye(LengthOptim) zeros(LengthOptim); %makes sure Pgrid in = 0 when Pgrid out != 0
         zeros(LengthOptim) eye(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) Pmax*eye(LengthOptim) zeros(LengthOptim);  %makes sure Pgrid out = 0 when Pgrid in != 0
         zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) eye(LengthOptim) -Pmax*eye(LengthOptim) zeros(LengthOptim); %makes sure X2  = 0 when X1 != 0
         zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) eye(LengthOptim) zeros(LengthOptim) Pmax*eye(LengthOptim) zeros(LengthOptim); %makes sure X1  = 0 when X2 != 0
         zeros(LengthOptim) zeros(LengthOptim) -tril(ones(LengthOptim)) tril(ones(LengthOptim)) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim); %makes sure SoC <= SoC max
         zeros(LengthOptim) zeros(LengthOptim) tril(ones(LengthOptim)) -tril(ones(LengthOptim)) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim); %makes sure SoC >= SoCmin
         zeros(LengthOptim) zeros(LengthOptim) eye(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) -Pbatmax*eye(LengthOptim) ; %makes sure Pbatdischarge  = 0 when Pcharge != 0
         zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) eye(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) zeros(LengthOptim) Pbatmax*eye(LengthOptim)]; %makes sure Pbat_charge  = 0 when Pdischarge != 0

    b = [zeros(LengthOptim,1); Pmax*ones(LengthOptim,1); zeros(LengthOptim,1); Pmax*ones(LengthOptim,1); (SoCmax-SoCInit)*ones(LengthOptim,1); (SoCInit - SoCmin)*ones(LengthOptim,1); zeros(LengthOptim,1); Pbatmax*ones(LengthOptim,1)]; 
    lb = [zeros(LengthOptim,1); zeros(LengthOptim,1); zeros(LengthOptim,1); zeros(LengthOptim,1); zeros(LengthOptim,1);  zeros(LengthOptim,1);  zeros(LengthOptim,1); zeros(LengthOptim,1)]  ;
    ub = [Pmax*ones(LengthOptim,1); Pmax*ones(LengthOptim,1); Pbatmax*ones(LengthOptim,1);  Pbatmax*ones(LengthOptim,1); Pmax*ones(LengthOptim,1);  Pmax*ones(LengthOptim,1);  ones(LengthOptim,1);  ones(LengthOptim,1) ]  ;

    %if we wanted to change the battery power for discharge for the night (to
    %follow a simulation where price was very high for selling and see if the simulation is able to find a good behaviour)
    %             ub(LengthOptim*2+21:LengthOptim*2+24)=3;
    % ub(LengthOptim*3+21:LengthOptim*3+24)=3;
    %             x0=[];
    %              options=optimoptions('intlinprog','Display','off');
    %             options.Display = 'off';
    [x,fval1,exitflag1,output1]  = intlinprog(f, Index_binary, A,b, Aeq, beq, lb, ub);
    SoC(i:i+LengthOptim-1) = -tril(ones(LengthOptim))*x(LengthOptim*2+1:LengthOptim*3)+tril(ones(LengthOptim))*x(LengthOptim*3+1:LengthOptim*4)+SoCInit*ones(LengthOptim,1);
    PgridIN = x(1:LengthOptim);
    PgridOUT = x(LengthOptim+1:LengthOptim*2);
    Pbat = x(LengthOptim*2+1:LengthOptim*3)-x(LengthOptim*3+1:LengthOptim*4);
    X1 = x(LengthOptim*4+1:LengthOptim*5);
    X2 = x(LengthOptim*5+1:LengthOptim*6);
    Alpha = x(LengthOptim*6+1:LengthOptim*7);
    Beta = x(LengthOptim*7+1:LengthOptim*8);

    SoCInit = SoC(i+LengthOptim-1);
    for j = 0:LengthOptim-1
        Energybought(i+j) = PgridIN(1+j)*BPrice(j+1);
        Energysold(i+j) =PgridOUT(1+j)*SPrice(j+1);          
        FromGrid(i+j) = max(0,(D(i+j)-P(i+j)-max(0,Pbat(j+1))*EfficiencyD+max(0,-Pbat(j+1)/EfficiencyC)));
        ToGrid(i+j) = max(0, (P(i+j)-D(i+j)+max(0,Pbat(j+1)*EfficiencyD)-max(0,-Pbat(j+1)/EfficiencyC)));
    end
    Powerbat (i:i+LengthOptim-1) = Pbat;
end

figure1 = figure('Name','Puissances','Color',[1 1 1]);
plot(time,P, time, D, time, Powerbat)
legend('Production', 'Demand', 'Battery')

figure2 = figure('Name','Prix','Color',[1 1 1]);
plot(time,BP, time, SP)
legend('Buying Price', 'Selling Price')
    
    
  
 
