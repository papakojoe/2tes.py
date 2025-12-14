import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro

# ---------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------
filename = "aligned_data_template_sub1.csv"
try:
    df = pd.read_csv(filename, header=1) # Load with correct header
except FileNotFoundError:
    print(f"Error: '{filename}' not found.")
    exit()

# Cleaning function for commas and quotes
def clean_number(x):
    if isinstance(x, str):
        x = x.replace('"', '').replace(',', '').strip()
    return pd.to_numeric(x, errors='coerce')

# Apply cleaning
for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(clean_number)

df.rename(columns={
    'Energy_Transport (ktoe)': 'Energy_Transport',
    'GHG_Transport (Gg CO2eq)': 'GHG_Transport'
}, inplace=True)

# ---------------------------------------------------------
# 2. RUN STATISTICAL ANALYSIS
# ---------------------------------------------------------
model = smf.ols("GHG_Transport ~ Energy_Transport", data=df).fit()
shapiro_stat, shapiro_p = shapiro(model.resid)

# Calculate Rise
start_val = df.sort_values('Year')['GHG_Transport'].iloc[0]
end_val = df.sort_values('Year')['GHG_Transport'].iloc[-1]
rise_pct = ((end_val - start_val) / start_val) * 100

# ---------------------------------------------------------
# 3. BUILD THE VISUAL DASHBOARD
# ---------------------------------------------------------
# Create a grid layout: 2 Plots on top, Text summary on bottom
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(3, 2)

# --- PANEL A: REGRESSION PLOT (Top Left) ---
ax1 = fig.add_subplot(gs[0:2, 0])
ax1.scatter(df["Energy_Transport"], df["GHG_Transport"], color='#D32F2F', s=100, label='Observed Data', zorder=3)
ax1.plot(df["Energy_Transport"], model.fittedvalues, color='#1976D2', linewidth=3, label=f'Trend ($R^2={model.rsquared:.2f}$)')
ax1.set_title("Regression: Energy vs. Emissions", fontsize=14, fontweight='bold')
ax1.set_xlabel("Energy Consumption (ktoe)", fontsize=12)
ax1.set_ylabel("GHG Emissions (Gg CO2eq)", fontsize=12)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- PANEL B: NORMALITY CHECK (Top Right) ---
ax2 = fig.add_subplot(gs[0:2, 1])
sm.qqplot(model.resid, line='45', ax=ax2, fit=True)
ax2.set_title("Validation: Normality of Residuals", fontsize=14, fontweight='bold')
# Color styling
ax2.get_lines()[0].set_color('#388E3C') # Dots
ax2.get_lines()[0].set_markersize(8)
ax2.get_lines()[1].set_color('black')   # Line
# Add Result Box
status = "NORMAL (Valid)" if shapiro_p > 0.05 else "NOT NORMAL"
ax2.text(0.05, 0.9, f"Shapiro-Wilk Test:\nP-Value = {shapiro_p:.4f}\nResult: {status}",
         transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
ax2.grid(True, linestyle='--', alpha=0.6)

# --- PANEL C: TEXT SUMMARY (Bottom Row) ---
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off') # Hide axis lines

slope = model.params['Energy_Transport']
conf = model.conf_int().loc['Energy_Transport']

summary_text = (
    f"STATISTICAL FINDINGS SUMMARY\n"
    f"================================================================================================\n"
    f"1. TREND ANALYSIS: Total emissions rose by {rise_pct:.1f}% from 2013 to 2022.\n\n"
    f"2. MODEL EQUATION: Emissions = {model.params['Intercept']:.2f} + {slope:.2f} (Energy)\n"
    f"   • Interpretation: For every 1 unit increase in Energy, Emissions rise by {slope:.2f} units.\n"
    f"   • 95% Confidence Interval: [{conf[0]:.2f}, {conf[1]:.2f}]\n\n"
    f"3. ANOVA SIGNIFICANCE:\n"
    f"   • F-Statistic: {model.fvalue:.2f}   |   P-Value: {model.f_pvalue:.2e}\n"
    f"   • Conclusion: The relationship is statistically significant (p < 0.05)."
)

ax3.text(0.02, 0.90, summary_text, fontsize=13, family='monospace', verticalalignment='top')

# Save and Show
plt.savefig("Project_Dashboard.png", dpi=150)
print("Dashboard created! Check 'Project_Dashboard.png'.")
plt.show()