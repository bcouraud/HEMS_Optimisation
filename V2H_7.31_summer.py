import pulp
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

T = 48  # 2day
hours = np.arange(T)

# Summer
load_1day = np.zeros(24)
load_1day += 1.0 / 24                       # Fridge
load_1day[9] += 1.0                         # Washing machine
load_1day[14] += 1.2                        # Clothes dryer
load_1day[19] += 1.1                        # Dishwasher
load_1day[7] += 0.3; load_1day[12] += 0.3; load_1day[17] += 0.3         # Kettle
load_1day[12] += 0.6; load_1day[18] += 0.6                        # Oven
load_1day[10] += 0.2; load_1day[15] += 0.2                        # Microwave
for t in range(18, 21): load_1day[t] += 0.3 / 3               # TV
for t in range(10, 18): load_1day[t] += 0.8 / 8               # Desktop
for t in [10,11,12,14,15,16,17,18]: load_1day[t] += 0.05     # Laptop
load_1day += 0.4 / 24                                        # Router/phone
load_1day[11] += 0.1                                         # Vacuum
for t in list(range(6,9)) + list(range(18,23)): load_1day[t] += 0.8 / 8  # Lighting
load_1day += 0.4 / 24                                        # Other
for t in range(9,19): load_1day[t] += 3.0 / 10                # AC

load = np.concatenate([load_1day, load_1day])

pv_1day = 1*np.array([
    0, 0, 0, 0, 0, 0.09325, 0.35755, 1.1788, 2.0436, 2.25335, 3.4553,
    4.1024, 4.15245, 4.29855, 3.93375, 3.328, 2.4308, 1.3769, 0.3791, 0.10215,
    0, 0, 0, 0
])
pv = np.concatenate([pv_1day, pv_1day])

# EV
battery_capacity = 78.0
soc_min = 0.2 * battery_capacity
soc_max = battery_capacity
charge_limit = 7
discharge_limit = 7

km_per_hour = 10
kwh_per_km = 0.1576
drive_use_per_hour = km_per_hour * kwh_per_km  # 3.94 kWh/h

# Electricity price
price_1day = np.zeros(24)
for t in range(24):
    if (t >= 23 or t < 7):
        price_1day[t] = 0.13897
    else:
        price_1day[t] = 0.29226
price = np.concatenate([price_1day, price_1day])
daily_fixed_charge = 0.6309
export_price = price * 0.50*0

# Ev away 2 day
ev_away_1day = list(range(8,12)) + list(range(14,18)) #+ list(range(19,21))
ev_away = ev_away_1day + [x+24 for x in ev_away_1day]

# model
model = pulp.LpProblem("KiaEV4_Summer_V2H_2days", pulp.LpMinimize)
charge = pulp.LpVariable.dicts("charge", hours, lowBound=0, upBound=charge_limit)
discharge = pulp.LpVariable.dicts("discharge", hours, lowBound=0, upBound=discharge_limit)
soc = pulp.LpVariable.dicts("soc", range(T+1), lowBound=soc_min, upBound=soc_max)
grid_import = pulp.LpVariable.dicts("grid_import", hours, lowBound=0)
grid_export = pulp.LpVariable.dicts("grid_export", hours, lowBound=0)
alpha = pulp.LpVariable.dicts("alpha", hours, 0, 1, cat="Binary")
beta = pulp.LpVariable.dicts("beta", hours, 0, 1, cat="Binary")
grid_limit = 100

model += (pulp.lpSum([grid_import[t]*price[t] - grid_export[t]*export_price[t] for t in hours]) + 2*daily_fixed_charge)

# Cycle
model += soc[0] == soc[T]

for t in hours:
    if t in ev_away:
        model += charge[t] == 0
        model += discharge[t] == 0
        model += soc[t+1] == soc[t] - drive_use_per_hour
    else:
        model += charge[t] <= charge_limit * beta[t]
        model += discharge[t] <= discharge_limit * (1 - beta[t])
        model += soc[t+1] == soc[t] + charge[t] - discharge[t]
    model += soc[t] <= battery_capacity
    model += soc[t] >= soc_min

model += soc[8] == battery_capacity
model += soc[32] == battery_capacity

for t in hours:
    model += grid_import[t] - grid_export[t] + discharge[t] - charge[t] + pv[t] - load[t] == 0
    model += grid_import[t] <= alpha[t] * grid_limit
    model += grid_export[t] <= (1 - alpha[t]) * grid_limit

model.solve()
print("status   =", pulp.LpStatus[model.status])

charge_values      = [charge[t].varValue for t in hours]
discharge_values   = [discharge[t].varValue for t in hours]
soc_values         = [soc[t].varValue for t in range(T+1)]
grid_import_values = [grid_import[t].varValue for t in hours]
grid_export_values = [grid_export[t].varValue for t in hours]

df = pd.DataFrame({
    "hour": hours,
    "charge": charge_values,
    "discharge": discharge_values,
    "soc": soc_values[:-1],
    "soc_next": soc_values[1:],
    "load": load,
    "pv": pv,
    "grid_import": grid_import_values,
    "grid_export": grid_export_values,
    "import_price": price,
    "export_price": export_price
})
df['grid_balance'] = df['grid_import'] - df['grid_export']

print("Hour\tImport\tExport\tNet\tSOC\tCharge\tDischarge\tLoad\tPV")
for t in hours:
    print(f"{t:02d}\t{df['grid_import'][t]:.2f}\t{df['grid_export'][t]:.2f}\t{df['grid_balance'][t]:.2f}\t"
          f"{df['soc'][t]:.2f}\t{df['charge'][t]:.2f}\t{df['discharge'][t]:.2f}\t{df['load'][t]:.2f}\t{df['pv'][t]:.2f}")

total_import = df['grid_import'].sum()
total_export = df['grid_export'].sum()
total_net = df['grid_balance'].sum()
print(f"\nTotal import: {total_import:.2f} kWh")
print(f"Total export: {total_export:.2f} kWh")
print(f"Net grid flow: {total_net:.2f} kWh")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=hours, y=df['charge'], mode='lines', name='Charge (kWh)', line=dict(color='green')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['discharge'], mode='lines', name='Discharge (kWh)', line=dict(color='red')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['load'], mode='lines', name='Load (kWh)', line=dict(color='blue')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['pv'], mode='lines', name='PV (kWh)', line=dict(color='orange')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['grid_import'], mode='lines', name='Grid Import (kWh)', line=dict(color='purple')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['grid_export'], mode='lines', name='Grid Export (kWh)', line=dict(color='brown', dash='dot')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['grid_balance'], mode='lines', name='Grid Net Flow (kWh)', line=dict(color='gray', dash='dash')), secondary_y=False)
fig.add_trace(go.Scatter(x=hours, y=df['soc'], mode='lines+markers', name='State of Charge (kWh)', line=dict(color='black', width=3)), secondary_y=True)
fig.update_layout(
    title="Kia EV4 V2H Household Energy Optimization (Physical Drive Out, Cyclic)",
    xaxis_title="Hour",
    legend=dict(orientation="h", y=-0.2),
)
fig.update_yaxes(title_text="Energy Flows (kWh)", secondary_y=False)
fig.update_yaxes(title_text="State of Charge (kWh)", secondary_y=True)
fig.show()

total_cost = pulp.value(model.objective)
print(f"\nTotal daily electricity cost: £{total_cost:.2f} (including standing charge)")