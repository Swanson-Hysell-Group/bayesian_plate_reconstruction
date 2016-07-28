import pandas as pd

pd.set_option('max_colwidth', 60)

df = pd.read_csv('pole_means.csv')
df.fillna('', inplace=True)
df.rename(columns={'PoleName': 'Pole',
                   'PLat': '$\\psi_p$', 
                   'PLon':'$\\phi_p$', 
                   'PmagRef': 'Pole reference',
                   'AgeNominal': 'Age',
                   'AgeLower': 'Lower age',
                   'AgeUpper': 'Upper age',
                   'AgeRef' : 'Age reference'}, inplace=True)
df = df[df.Pole != 'Osler_N'] #Huge error, does not contribute much to the model
df = df[df.Pole != 'Abitibi'] # Standstill at the beginning, not realistic to fit
df = df[df.Pole != 'Haliburton'] #Much younger, far away pole, difficutlt to fit

with open("keweenawan_poles.tex", 'w') as f:
    df[['Pole', '$\\psi_p$', '$\\phi_p$', 'A95', 'Pole reference',\
        'Age', 'Lower age', 'Upper age', 'Age reference']].to_latex(f, 
         escape=False, longtable=False, index=False,
         column_format='p{3cm} p{1.0cm} p{1.0cm} p{1.0cm} p{4cm} p{2.0cm} p{1.5cm} p{1.5cm} p{4cm}')
