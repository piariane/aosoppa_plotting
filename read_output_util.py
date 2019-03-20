import numpy as np

def read_input(sing_trip, method, molecule, sym, state, RPA_imag):

    # Which file:
    if (sing_trip == 'singlet'):
        outputfile = '%s/aosoppa_all_sing_%s.out' % (molecule,molecule)
    if (sing_trip == 'triplet'):
        outputfile = '%s/aosoppa_all_trip_%s.out' % (molecule,molecule)
    f = open(outputfile, 'r')

    # What to search for
    pattern1a = '%s iterations, Excitation symmetry %d' % (method,sym)
    pattern1b = '%s calculation, Excitation symmetry %d' % (method,sym)
    pattern2 = 'Excitation symmetry  %d, state no.  %d' % (sym, state)
    pattern3 = '%s excitation energy' % (method)
    pattern4 = 'The irrep name for each symmetry'
    pattern5 = 'Point group:'

    method_has_doubles = False
    # Start loop over lines in output:
    while True:
        line = f.readline()

        if (pattern4) in line:

            sym_index = int(7+2*sym)
            symmetry_label = line.split()[sym_index]
#            if (method == 'SOPPA'):
#                print symmetry_label

        if (pattern5) in line:

            if (method == 'SOPPA'):
                point_group = line.split()[3]
#                print point_group

        if (pattern2) in line:
            newline = f.readline()
            newline = f.readline()

            if pattern3 in newline:

                # Start loop over methods
                for line2 in f:
        
                    # Get excitation energy (eV)
                    if 'eV' in line2:
                        single_exci_method = []
                        ev = line2.split()[0]
        
                    # Get 2p-2h excitation weight
                    if '2p-2h + 2h-2p excitation weight' in line2:
                        phph_weight = line2.split()[-2]
        
                    # Get 1p-1h excitation weight
                    if '1p-1h + 1h-1p excitation weight' in line2:
                        ph_weight = line2.split()[-2]
        
                    # Get ph eigenvector
                    if 'exci.' in line2:
                        ph = [line2.split()[3], line2.split()[4], line2.split()[6],
                              line2.split()[7], line2.split()[9], 0]
                        single_exci_method.append(ph)
        
                    # Go to double excitation part
                    if '-----' in line2:
                        double_exci_method = []
                        for line3 in f:
        
                            # Get eigenvector elemente with weigth larger than
                            if 'exci.' in line3:
                                phph = [line3.split()[3], line3.split()[4], line3.split()[5],
                                        line3.split()[6], line3.split()[8], line3.split()[9],
                                        line3.split()[10], line3.split()[11],
                                        line3.split()[13], 0]
                                double_exci_method.append(phph)
                                method_has_doubles = True
        
                            # Stop iteration over double excitation part
                            if 'Printed all' in line3:
                                if not method_has_doubles:
                                    phph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                    double_exci_method.append(phph)
                                break
                        break
                break
#        break
    # Don't include imaginary RPA energies
    if (float(ev) > 0.1):
        single_exci = np.array(single_exci_method, dtype=float)
        if (method == 'RPA(D)') and RPA_imag:
            single_exci = np.zeros((1,6), dtype=float)
        RPA_imag = False
    
    # If imaginary RPA energies, insert zeros
    else:
        single_exci = np.zeros((1,6), dtype=float)
        RPA_imag = True
    
    # Create dictionaries
    double_exci = np.array(double_exci_method, dtype=float)
    energy_exci = np.array([[ev, ph_weight, 0]], dtype=float)

    f.close()

    return single_exci, double_exci, energy_exci, RPA_imag, symmetry_label

def get_state(sing_trip, methods, molecule, sym, state):

    single_exci = dict()
    double_exci = dict()
    energy_exci = dict()

    imag = False
    for a in range(len(methods)):

        method = methods[a]
        statea = state[a]

        single_exci[method], double_exci[method], energy_exci[method], imag, sym_index = read_input(sing_trip, method, molecule, sym, statea, imag)

    single_max_len = max(len(single_exci[i]) for i in single_exci)
    double_max_len = max(len(double_exci[i]) for i in double_exci)

    # Initialize 'eigenvector' arrays
    single_1 = np.zeros((single_max_len,1))
    double_1 = np.zeros((double_max_len,1))

    # Initialize array for energies and weigths to display with eigenvectors
    energy_1 = np.zeros((single_max_len,1))

    # Initialize array for energies only
    energy_tot = np.zeros((1,1))

    # Initialize array for weigths only
    weight_tot = np.zeros((1,1))

    mol_name = np.zeros((1,3), dtype=object)
    mol_name[0,0] = molecule
    mol_name[0,1] = state[-1]
    mol_name[0,2] = sym_index

    mol_padded = np.pad(mol_name, ((0,single_max_len-len(mol_name)),(0,0)),
                                    'constant', constant_values=0)

    for i in methods:

        # single and double eigenvector elements
        single_meth = single_exci[i]
        double_meth = double_exci[i]

        # energy_exci = np.array([[ev, ph_weight, 0]], dtype=float)
        energy_meth = energy_exci[i]

        # pad arrays with zero so they will fir in total_1 and total_2
        single_padded = np.pad(single_meth, ((0, single_max_len-len(single_meth)), (0,0)),
                               'constant', constant_values=0)
        energy_padded = np.pad(energy_meth, ((0,single_max_len-1),(0,0)),
                               'constant', constant_values=0)
        double_padded = np.pad(double_meth, ((0,double_max_len-len(double_meth)),(0,0)),
                               'constant', constant_values=0)

        # concatenate methods
        single_1 = np.concatenate((single_1, single_padded), axis=1)
        energy_1 = np.concatenate((energy_1, energy_padded), axis=1)
        double_1 = np.concatenate((double_1, double_padded), axis=1)

        energy_meth_1 = np.expand_dims(np.array([energy_meth[0,0]]),axis=1)
        energy_tot = np.concatenate((energy_tot, energy_meth_1), axis=1)

        weight_meth_1 = np.expand_dims(np.array([energy_meth[0,1]]),axis=1)
        weight_tot = np.concatenate((weight_tot, weight_meth_1), axis=1)

    # concatenate molecule info, energy and weights and eigenvectors
    single_1 = np.concatenate((mol_padded,energy_1, single_1), axis=1)
    energy_tot = np.concatenate((mol_name, energy_tot), axis=1)
    weight_tot = np.concatenate((mol_name, weight_tot), axis=1)

    return single_1, double_1, energy_tot, weight_tot

def create_table(sing_trip, methods, sym, state, molecule, j):

    max_len_1 = 5 + (3*len(methods)) + (6*len(methods))
    max_len_2 = 1 + (10*len(methods))

    total_1 = np.zeros((1,max_len_1))
    total_2 = np.zeros((1,max_len_2))
    extra_1 = np.zeros((1,max_len_1))
    extra_2 = np.zeros((1,max_len_2))

    total_energy = np.zeros((1,4+len(methods)), dtype=object)
    total_weight = np.zeros((1,4+len(methods)), dtype=object)
    total_rpadenergy = np.zeros((1,5), dtype=object)
    total_hrpadenergy = np.zeros((1,5), dtype=object)
    total_shrpadenergy = np.zeros((1,5), dtype=object)

    # Calculate and collect all states in the same array
    for i in range(j):

        print(molecule[i], sym[i], state[i])

        single_1, single_2, energy_tot, weight_tot  = get_state(sing_trip, methods, molecule[i], sym[i], state[i,:])

        rpad_energy =  read_double_corrections('RPA(D)', sing_trip, molecule[i], sym[i], state[i,:])
        hrpad_energy = read_double_corrections('HRPA(D)', sing_trip, molecule[i], sym[i], state[i,:])
        shrpad_energy = read_double_corrections('s-HRPA(D)', sing_trip, molecule[i], sym[i], state[i,:])

        total_1 = np.concatenate((total_1, single_1), axis=0)
        total_1 = np.concatenate((total_1, extra_1), axis=0)
        total_2 = np.concatenate((total_2, single_2), axis=0)
        total_2 = np.concatenate((total_2, extra_2), axis=0)

        total_energy = np.concatenate((total_energy, energy_tot), axis=0)

        total_weight = np.concatenate((total_weight, weight_tot), axis=0)

        total_rpadenergy = np.concatenate((total_rpadenergy, rpad_energy), axis=0)
        total_rpadenergy[i+1,0] = molecule[i]

        total_hrpadenergy = np.concatenate((total_hrpadenergy, hrpad_energy), axis=0)
        total_hrpadenergy[i+1,0] = molecule[i]

        total_shrpadenergy = np.concatenate((total_shrpadenergy, shrpad_energy), axis=0)
        total_shrpadenergy[i+1,0] = molecule[i]

    # Remove original zero line
    total_energy = np.delete(total_energy, 0, 0)
    total_energy = np.delete(total_energy, 3, 1)
    total_weight = np.delete(total_weight, 0, 0)
    total_weight = np.delete(total_weight, 3, 1)
    total_rpadenergy = np.delete(total_rpadenergy, 0, 0)
    total_hrpadenergy = np.delete(total_hrpadenergy, 0, 0)
    total_shrpadenergy = np.delete(total_shrpadenergy, 0, 0)

#    # Save 'eigenvectors'
    np.savetxt('total_single_3', total_1, fmt='%s')
    np.savetxt('total_double_3', total_2, fmt='%s')
#
#    # Save all excitation energies
    np.savetxt('total_energy_3', total_energy, fmt='%-16s %6i %6s'+'%8.2f'*len(methods))
    np.savetxt('total_weight_3', total_weight, fmt='%-16s %6i %6s'+'%8.2f'*len(methods))
#
#    # Save RPA(D) and HRPA(D) excitation contributions
    np.savetxt('rpad_exci_contribution_3', total_rpadenergy, fmt='%s')
    np.savetxt('hrpad_exci_contribution_3', total_hrpadenergy, fmt='%s')
    np.savetxt('shrpad_exci_contribution_3', total_hrpadenergy, fmt='%s')

def read_double_corrections(method, sing_trip, molecule, sym, state):

    # Which file:
    if (sing_trip == 'singlet'):
        outputfile = '%s/aosoppa_all_sing_%s.out' % (molecule,molecule)
    if (sing_trip == 'triplet'):
        outputfile = '%s/aosoppa_all_trip_%s.out' % (molecule,molecule)
    f = open(outputfile, 'r')

    # What to search for
    pattern1 = method+' calculation, Excitation symmetry %d' % (sym)
#    pattern2 = 'HRPA(D) calculation, Excitation symmetry %d' % (sym)
#    pattern2 = 's-HRPA(D) calculation, Excitation symmetry %d' % (sym)

    # states
    if method == 'RPA(D)':
        state1 = state[1]
    if method == 'HRPA(D)':
        state1 = state[3]
    if method == 's-HRPA(D)':
        state1 = state[4]
#    state2 = state[3]
#    state2 = state[4]

    # Initiate arrays
    rpad_energies = np.zeros((1,5), dtype=object)
#    hrpad_energies = np.zeros((1,5), dtype=object)

    # Start loop over lines in output:
    for m,line in enumerate(f):

        if pattern1 in line:

            # Start loop over methods
            for n,line2 in enumerate(f):

                if 'Excitation' in line2:

                    for line3 in f:

                        if '-----' in line3:

                            for line4 in f:

                                if '-----' in line4:
                                    break

                                if (float(line4.split()[0]) == state1):

                                    rpad_energies[0,1] = line4.split()[1]
                                    rpad_energies[0,2] = line4.split()[2]
                                    rpad_energies[0,3] = line4.split()[3]
                                    rpad_energies[0,4] = line4.split()[4]

                                    break
                            break
                    break
            break

    return rpad_energies
