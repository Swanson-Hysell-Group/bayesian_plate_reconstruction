import pandas as pd
import numpy as np

pd.set_option('max_colwidth', 60)

df = pd.read_csv('GPMDB_Australia_50Ma.csv')
df.fillna('', inplace=True)

df.rename(columns={'ROCKNAME': 'Pole',
                   'PLAT': '$\\psi_p$', 
                   'PLONG':'$\\phi_p$', 
                   'LOMAGAGE': 'Lower age (Ma)',
                   'HIMAGAGE': 'Upper age (Ma)'}, inplace=True)

df.sort_values(by='Upper age (Ma)', inplace=True)

df['$A_{95}$'] = np.round(np.sqrt(df['DP']*df['DM']), decimals=1)

with open("australia_poles.tex", 'w') as f:
    df[['Pole', '$\\psi_p$', '$\\phi_p$', '$A_{95}$', 'Reference',\
        'Lower age (Ma)', 'Upper age (Ma)']].to_latex(f, 
         float_format="%g", escape=False, longtable=False, index=False,
         column_format='p{4cm} p{1.0cm} p{1.0cm} p{1.0cm} p{2.5cm} p{1cm} p{1cm}')
