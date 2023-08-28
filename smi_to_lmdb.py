import os
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import random
import argparse
from tqdm import tqdm
SEED = 0
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED)


def smi2coords(smi, seed):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinate_list = []
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        coordinates = mol.GetConformer().GetPositions()
    elif res == -1:
        mol_tmp = Chem.MolFromSmiles(smi)
        AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_tmp)
        except:
            pass
        coordinates = mol_tmp.GetConformer().GetPositions()
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    coordinate_list.append(coordinates.astype(np.float32))
    return pickle.dumps({'atoms': atoms, 'coordinates': coordinate_list, 'smi': smi}, protocol=-1)



def write_lmdb(smiles_list, outpath, seed=42):

    env_new = lmdb.open(
        outpath,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    ignore_num = 0
    for i, smiles in tqdm(enumerate(smiles_list)):
        try:
            inner_output = smi2coords(smiles, seed=seed)
        except:
            print(f"{smiles} get coords error, this program will skip this smile!!!!")
            ignore_num += 1
            continue

        if inner_output is None:
            print(f"{smiles} get coords none, this program will skip this smile!!!!")
            ignore_num += 1
            continue

        txn_write.put(f"{i-ignore_num}".encode("ascii"), inner_output)
    txn_write.commit()
    env_new.close()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default='example.smi', type=str)
    parser.add_argument('-o', '--output_path', default='example.lmdb', type=str)
    args = parser.parse_args()

    print(args)
    smi_list_df = pd.read_csv(args.input_path, sep=" ", header=None)
    smi_list = smi_list_df.iloc[:,0].tolist()
    write_lmdb(smi_list, args.output_path, seed=SEED)







# CUDA_VISIBLE_DEVICES=0 python Uni-Mol/unimol/unimol/infer.py example.lmdb --results-path example.pkl \ 
# --user-dir Uni-Mol/unimol/unimol --path /Arontier_2/Projects/rna_binder/230804/pretrained_unimol/mol_pre_no_h_220816.pt \
# --num-workers 8 --ddp-backend=c10d --batch-size 32 --task unimol --loss unimol_infer --arch unimol_base \ 
# --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --only-polar 0 --dict-name dict.txt --log-interval 50 \ 
# --log-format simple --random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer

