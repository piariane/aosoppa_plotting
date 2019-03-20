import numpy as np
import sys
sys.path.insert(0, '/home/zth702/data/Masterthesis2016/aosoppa_triplet/')
from read_output_util import *


methods = ['RPA', 'RPA(D)', 'Higher RPA', 'HRPA(D)', 's-HRPA(D)', 'SOPPA', 'SOPPA(CC2)', 'SOPPA(CCSD)']

molecule = np.loadtxt('molecules.txt', usecols=[0], dtype=str)
sym = np.loadtxt('molecules.txt', usecols=[1])
states = np.loadtxt('molecules.txt', usecols=range(2,2+len(methods),1))

create_table('triplet', methods, sym, states, molecule, len(molecule))
