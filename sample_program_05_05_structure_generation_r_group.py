# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd
from rdkit import Chem

number_of_generating_structures = 10000  # 生成する化学構造の数

dataset_main_fragments = pd.read_csv('main_fragments.csv', index_col=0)  # 主骨格のフラグメントの SMILES のデータセットの読み込み
main_fragments = list(dataset_main_fragments.iloc[:, 0])  # 主骨格のフラグメントの SMILES
dataset_sub_fragments = pd.read_csv('sub_fragments.csv', index_col=0)  # 側鎖のフラグメントの SMILES のデータセットの読み込み
sub_fragments = list(dataset_sub_fragments.iloc[:, 0])  # 側鎖のフラグメントの SMILES

print('主骨格のフラグメントの数 :', len(main_fragments))
print('側鎖のフラグメントの数 :', len(sub_fragments))

main_molecules = [Chem.MolFromSmiles(smiles) for smiles in main_fragments]
fragment_molecules = [Chem.MolFromSmiles(smiles) for smiles in sub_fragments]
    
bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
             Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
             Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
             Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
             Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
             Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
             Chem.rdchem.BondType.ZERO]

generated_structures = []
for generated_structure_number in range(number_of_generating_structures):
    selected_main_molecule_number = np.floor(np.random.rand(1) * len(main_molecules)).astype(int)[0]
    main_molecule = main_molecules[selected_main_molecule_number]
    # make adjacency matrix and get atoms for main molecule
    main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(main_molecule)
    for bond in main_molecule.GetBonds():
        main_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
        main_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())
    main_atoms = []
    for atom in main_molecule.GetAtoms():
        main_atoms.append(atom.GetSymbol())
    
    r_index_in_main_molecule_old = [index for index, atom in enumerate(main_atoms) if atom == '*']
    for index, r_index in enumerate(r_index_in_main_molecule_old):
        modified_index = r_index - index
        atom = main_atoms.pop(modified_index)
        main_atoms.append(atom)
        tmp = main_adjacency_matrix[:, modified_index:modified_index + 1].copy()
        main_adjacency_matrix = np.delete(main_adjacency_matrix, modified_index, 1)
        main_adjacency_matrix = np.c_[main_adjacency_matrix, tmp]
        tmp = main_adjacency_matrix[modified_index:modified_index + 1, :].copy()
        main_adjacency_matrix = np.delete(main_adjacency_matrix, modified_index, 0)
        main_adjacency_matrix = np.r_[main_adjacency_matrix, tmp]
    r_index_in_main_molecule_new = [index for index, atom in enumerate(main_atoms) if atom == '*']
    
    r_bonded_atom_index_in_main_molecule = []
    for number in r_index_in_main_molecule_new:
        r_bonded_atom_index_in_main_molecule.append(np.where(main_adjacency_matrix[number, :] != 0)[0][0])
    r_bond_number_in_main_molecule = main_adjacency_matrix[
        r_index_in_main_molecule_new, r_bonded_atom_index_in_main_molecule]
    
    main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 0)
    main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 1)
    
    for i in range(len(r_index_in_main_molecule_new)):
        main_atoms.remove('*')
    main_size = main_adjacency_matrix.shape[0]
    
    selected_fragment_numbers = np.floor(np.random.rand(len(r_index_in_main_molecule_old)) * len(fragment_molecules)).astype(int)
      
    generated_molecule_atoms = main_atoms[:]
    generated_adjacency_matrix = main_adjacency_matrix.copy()
    for r_number_in_molecule in range(len(r_index_in_main_molecule_new)):
        fragment_molecule = fragment_molecules[selected_fragment_numbers[r_number_in_molecule]]
        fragment_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(fragment_molecule)
        for bond in fragment_molecule.GetBonds():
            fragment_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(
                bond.GetBondType())
            fragment_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(
                bond.GetBondType())
        fragment_atoms = []
        for atom in fragment_molecule.GetAtoms():
            fragment_atoms.append(atom.GetSymbol())

        # integrate adjacency matrix
        r_index_in_fragment_molecule = fragment_atoms.index('*')

        r_bonded_atom_index_in_fragment_molecule = \
            np.where(fragment_adjacency_matrix[r_index_in_fragment_molecule, :] != 0)[0][0]
        if r_bonded_atom_index_in_fragment_molecule > r_index_in_fragment_molecule:
            r_bonded_atom_index_in_fragment_molecule -= 1

        fragment_atoms.remove('*')
        fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 0)
        fragment_adjacency_matrix = np.delete(fragment_adjacency_matrix, r_index_in_fragment_molecule, 1)
    
        main_size = generated_adjacency_matrix.shape[0]
        generated_adjacency_matrix = np.c_[generated_adjacency_matrix, np.zeros(
            [generated_adjacency_matrix.shape[0], fragment_adjacency_matrix.shape[0]], dtype='int32')]
        generated_adjacency_matrix = np.r_[generated_adjacency_matrix, np.zeros(
            [fragment_adjacency_matrix.shape[0], generated_adjacency_matrix.shape[1]], dtype='int32')]

        generated_adjacency_matrix[r_bonded_atom_index_in_main_molecule[
                                       r_number_in_molecule], r_bonded_atom_index_in_fragment_molecule + main_size] = \
            r_bond_number_in_main_molecule[r_number_in_molecule]
        generated_adjacency_matrix[
            r_bonded_atom_index_in_fragment_molecule + main_size, r_bonded_atom_index_in_main_molecule[
                r_number_in_molecule]] = r_bond_number_in_main_molecule[r_number_in_molecule]
        generated_adjacency_matrix[main_size:, main_size:] = fragment_adjacency_matrix

        # integrate atoms
        generated_molecule_atoms += fragment_atoms

    # generate structures 
    generated_molecule = Chem.RWMol()
    atom_index = []
    for atom_number in range(len(generated_molecule_atoms)):
        atom = Chem.Atom(generated_molecule_atoms[atom_number])
        molecular_index = generated_molecule.AddAtom(atom)
        atom_index.append(molecular_index)
    for index_x, row_vector in enumerate(generated_adjacency_matrix):
        for index_y, bond in enumerate(row_vector):
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            else:
                generated_molecule.AddBond(atom_index[index_x], atom_index[index_y], bond_list[bond])

    generated_molecule = generated_molecule.GetMol()
    generated_structures.append(Chem.MolToSmiles(generated_molecule))
    if (generated_structure_number + 1) % 1000 == 0 or (generated_structure_number + 1) == number_of_generating_structures:
        print(generated_structure_number + 1, '/', number_of_generating_structures)
generated_structures = list(set(generated_structures))  # 重複する構造の削除
generated_structures = pd.DataFrame(generated_structures, columns=['SMILES'])
generated_structures.to_csv('generated_structures_r_group.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
