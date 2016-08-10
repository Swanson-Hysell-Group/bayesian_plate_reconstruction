import pandas as pd

pd.set_option('max_colwidth', 60)

df = pd.read_csv('pole_means.csv')
df.fillna('', inplace=True)
df.rename(columns={'PoleName': 'Pole',
                   'PLat': '$\\psi_p$', 
                   'PLon':'$\\phi_p$', 
                   'A95': '$A_{95}$',
                   'PmagRef': 'Pole reference',
                   'AgeNominal': 'Age (Ma)',
                   'AgeLower': 'Lower age (Ma)',
                   'AgeUpper': 'Upper age (Ma)',
                   'AgeRef' : 'Age reference'}, inplace=True)
df = df[df.Pole != 'Osler normal (upper)'] #Huge error, does not contribute much to the model
df = df[df.Pole != 'Abitibi'] # Standstill at the beginning, not realistic to fit
df = df[df.Pole != 'Haliburton'] #Much younger, far away pole, difficutlt to fit

with open("keweenawan_poles.tex", 'w') as f:
    df[['Pole', '$\\psi_p$', '$\\phi_p$', '$A_{95}$', 'Pole reference',\
        'Age (Ma)', 'Lower age (Ma)', 'Upper age (Ma)', 'Age reference']].to_latex(f, 
         escape=False, longtable=False, index=False,
         column_format='p{3cm} p{0.8cm} p{0.8cm} p{0.8cm} p{4cm} p{2.0cm} p{1.2cm} p{1.2cm} p{4.0cm}')
