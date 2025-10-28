import numpy as np
import matplotlib.pyplot as plt

# One day in minutes
minutes_per_day = 96
time = np.arange(minutes_per_day)  # 0 to 1439

# --- Solar Production Profile ---
# Model: A sine curve peaking at noon, zero at night
# We'll simulate sunrise at 6:00 (360 min) and sunset at 18:00 (1080 min)
solar = np.zeros(minutes_per_day)
daylight = (time >= 360/1440*96) & (time <= 1080/1440*96)
solar[daylight] = np.sin(np.pi * (time[daylight] - 360/1440*96) / (1080/1440*96 - 360/1440*96))
solar = np.clip(solar, 0, None)  # no negative values
solar *= 5  # peak production (e.g., 5 kW)

# --- Household Consumption Profile ---
# Model: higher morning (6–9h) and evening (18–22h) usage, baseline otherwise
base_load = 0.3 + 0.1 * np.random.randn(minutes_per_day)  # small random variation
morning_peak = ((time >= 360/1440*96) & (time <= 540/1440*96)) * (0.5 + 0.1 * np.random.randn(minutes_per_day))   # 6–9h
evening_peak = ((time >= 1080/1440*96) & (time <= 1320/1440*96)) * (0.6 + 0.1 * np.random.randn(minutes_per_day)) # 18–22h
consumption = base_load + morning_peak + evening_peak
consumption = np.clip(consumption, 0.1, None)  # avoid negative consumption

# --- Optional: Plot ---
plt.figure(figsize=(10,5))
plt.plot(time / 60, solar, label="Solar Production (kW)")
plt.plot(time / 60, consumption, label="Household Consumption (kW)")
plt.xlabel("Time (hours)")
plt.ylabel("Power (kW)")
plt.title("Simulated Daily Solar Production and Household Consumption")
plt.legend()
plt.grid(True)
plt.show()

# Both arrays are now ready for use
print("Solar production array shape:", solar.shape)
print("Consumption array shape:", consumption.shape)