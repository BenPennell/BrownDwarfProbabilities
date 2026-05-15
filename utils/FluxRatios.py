import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

'''
    Script written by Jiadong
'''

'''
    CHOOSE THE IMPORT FILE, FLUX RATIO, AND OUTDIR HERE
'''
# load in file
df = pd.read_csv('./data/PARSEC_logage_6to10_MH_n2p5_0p6_dwarf.csv')
# Target flux ratio
flux_ratio_target = 10**(-1.5)
# outdir
outdir = './data/f_003_many_metallicities.csv'
'''
    CHOOSE THE IMPORT FILE, FLUX RATIO, AND OUTDIR HERE
'''

# For secondary to have flux_ratio = 10**(-1.5) relative to primary:
# Gmag_2 - Gmag_1 = -2.5 * log10(10**(-1.5))
delta_mag = -2.5 * np.log10(flux_ratio_target)

# Use logAge = 9.0 (1 Gyr) for main sequence stars
target_age = 9.0
df_age = df[df['logAge'] == target_age].copy()

# Get unique metallicities
mh_values = sorted(df_age['MH'].unique())

# Sample [M/H] values for the table
mh_sample = np.linspace(-0.5,0.5,25) #[-2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5]

# Sample primary masses (Mini_1)
mini_sample = np.arange(0.02,0.8,0.01) #np.arange(0.3,0.85,0.01)
mini_sample = np.round(mini_sample, 2)

results = []

for mh in mh_sample:
    # Get closest available MH
    mh_closest = min(mh_values, key=lambda x: abs(x - mh))
    df_mh = df_age[df_age['MH'] == mh_closest].copy()
    
    if len(df_mh) == 0:
        continue
    
    for mini_1 in mini_sample:
        df_mh_sorted = df_mh.sort_values('Mini')
        
        # Get available masses and Gmag
        mini_available = df_mh_sorted['Mini'].values
        gmag_available = df_mh_sorted['Gmag'].values
        
        if mini_1 < mini_available.min() or mini_1 > mini_available.max():
            #print("B")
            continue
            
        try:
            # Interpolate to get Gmag at mini_1
            interp_gmag = interp1d(mini_available, gmag_available, kind='linear')
            Gmag_1 = float(interp_gmag(mini_1))
            
            # The secondary needs Gmag_2 = Gmag_1 + delta_mag (fainter)
            Gmag_2_target = Gmag_1 + delta_mag
            
            # Find mass range where star is fainter (lower mass)
            mask = mini_available < mini_1
            if not np.any(mask):
                continue
                
            mini_subset = mini_available[mask]
            gmag_subset = gmag_available[mask]
            
            # Check if target Gmag is in range
            # if Gmag_2_target < gmag_subset.min() or Gmag_2_target > gmag_subset.max():
            #     print(mini_1, "uh oh")
            #     continue
            
            # Sort by Gmag for interpolation
            sort_idx = np.argsort(gmag_subset)
            gmag_sorted = gmag_subset[sort_idx]
            mini_sorted = mini_subset[sort_idx]
            
            # Interpolate mass from Gmag
            interp_mass = interp1d(gmag_sorted, mini_sorted, kind='linear')
            mini_2 = float(interp_mass(Gmag_2_target))
            
            mass_ratio = mini_2 / mini_1
            #if 0 < mass_ratio < 1:
            results.append({
                '[M/H]': round(mh_closest, 2),
                'Mini': mini_1,
                'Mini_2': round(mini_2, 4),
                'mass_ratio': round(mass_ratio, 4)
                })
        except Exception as e:
            #print("D")
            pass

# Create results table
results_df = pd.DataFrame(results)

print("="*80)
print("MASS RATIO THRESHOLD TABLE")
print(f"Flux ratio threshold: {flux_ratio_target}")
print(f"Condition: flux_2 / flux_1 = 10**(-1.5) (secondary is 10x fainter)")
print(f"delta_Gmag = {delta_mag:.2f} mag")
print(f"Age: logAge = {target_age} (1 Gyr)")
print("="*80)
print()
print("mass_ratio = Mini_2 / Mini_1")
print("(where Mini_2 is the secondary mass that produces flux_ratio = 10**(-1.5))")
print()

# Create pivot table
if len(results_df) > 0:
    pivot = results_df.pivot_table(values='mass_ratio', index='Mini', columns='[M/H]', aggfunc='first')
    
    # Format nicely
    print("Mass Ratio (q = Mini_2/Mini_1) vs Primary Mass and Metallicity:")
    print("-"*80)
    
    # Print header
    header = "Mini " + " ".join([f"{mh:>7}" for mh in pivot.columns])
    print(header)
    print("-"*80)
    
    # Print rows
    for mini in pivot.index:
        row = f"{mini:>4.1f}"
        for mh in pivot.columns:
            val = pivot.loc[mini, mh]
            if pd.isna(val):
                row += "     N/A"
            else:
                row += f"  {val:>5.3f}"
        print(row)
    
    print("-"*80)
    print()
    
    # Also save as CSV
    pivot.to_csv(outdir)
    print("Table saved to: " + outdir)
    
    # Print interpretation
    print()
    print("INTERPRETATION:")
    print("-"*80)
    print("For a binary with flux_ratio = 10**(-1.5) (secondary 10x fainter than primary),")
    print("the mass ratio threshold is given above.")
    print()
    print("Key obsemini_available = df_mh_sorted['Mini'].valuesrvations:")
    print("- Higher metallicity → slightly higher mass ratio threshold")
    print("- Mass ratio ranges from ~0.35 (low mass, metal-poor) to ~0.7 (solar mass)")
    print("- For a 1 M☉ primary at [M/H]=0, q ≈ 0.67 means Mini_2 ≈ 0.67 M☉")
    mini_available = df_mh_sorted['Mini'].values
    print(mini_available)