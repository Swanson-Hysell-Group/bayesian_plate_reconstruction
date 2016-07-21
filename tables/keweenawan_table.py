import pandas as pd

pd.set_option('max_colwidth', 60)

df = pd.read_csv('pole_means.csv')
df.fillna('', inplace=True)
df.rename(columns={'PLat': '$\\psi_p$', 
                   'PLon':'$\\phi_p$', 
                   'A95': '$\\alpha_{95}$', 
                   'PmagRef': 'Pole reference',
                   'AgeNominal': 'Nominal age',
                   'AgeLower': 'Lower age',
                   'AgeUpper': 'Upper age',
                   'gaussian2sigma': '$2\\sigma$',
                   'AgeRef' : 'Age reference'}, inplace=True)

with open("keweenawan_poles.tex", 'w') as f:
    df[['Pole name', '$\\psi_p$', '$\\phi_p$', '$\\alpha_{95}$', 'Pole reference',\
        'Nominal age', 'Lower age', 'Upper age', 'Age reference']].to_latex(f, escape=False, longtable=True, index=False)
