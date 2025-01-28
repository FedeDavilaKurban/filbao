from nbodykit.lab import *
import pandas as pd
import numpy as np

zmin, zmax = 0.01, 0.15

# Load your data and randoms as Pandas DataFrames
randoms = pd.read_csv('../data/random_sample_healpy_128_4478782.csv')  # Replace with your randoms CSV file path

# Convert DataFrames to ArrayCatalog with RA, DEC, and Redshift
rand_cat = ArrayCatalog({'RA': randoms['ra'].values,
                         'DEC': randoms['dec'].values,
                         'Redshift': randoms['red'].values})


# Assign cosmology (if not already in the data)
cosmo = cosmology.Planck15

# Define the edges for s and mu bins
s_bins = np.linspace(30,150,100)  # smaller range for simplicity
mu_bins = len(s_bins)-1  # fewer bins for testing

# Define the SurveyData2PCF object
result = SurveyDataPairCount('2d', rand_cat, edges=s_bins, Nmu=mu_bins, cosmo=cosmo, ra='RA', dec='DEC', redshift='Redshift')

result.save('RRpair.json')
print('JSON file created')