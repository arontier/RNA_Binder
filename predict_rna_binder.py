import os
import pandas as pd
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
import argparse
import glob

SEED = 0
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RNA_CLASS_GROUP_LABEL = {
    "G4": 0,
    "Hairpin": 1,
    "PK": 2,
    "TWJ": 3,
    "TH": 4,
}
RNA_CLASS_GROUP = {
    "G4": ["AKTIP", "EWSR1", "NRAS", "TERRA", "Zika3PrimeUTR", "Zika_NS5", ],
    "Hairpin": ["BCL_XL", "BCL_XL_SS", "FGFR", "HBV", "HIV_SL3", "KLF6_mut", "KLF6_wt", "Pre_miR_17", "Pre_miR_21",
                "Pre_miR_31", "Pro_mut", "Pro_wt", "RRE2B_MeA", "RRE2B", ],
    "PK": ["PreQ1", "SAM_ll", "ZTP", ],
    "TWJ": ["Glutamine_RS", "TPP", ],
    "TH": ["ENE_A9", "MALAT1", ],
}


class rna_bind_mlp(torch.nn.Module):
    def __init__(self, class_num=5, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(dropout)

        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, 512),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 )

        self.mlp_binary_out = nn.Sequential(nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(128, class_num),
                                            )

    def forward(self, mol_feat):
        mol_feat = self.mlp(mol_feat)
        binary_logits = self.mlp_binary_out(mol_feat)

        return binary_logits


class rna_bind_dataset(torch.utils.data.Dataset):
    def __init__(self, data_smile, data_x):
        super(rna_bind_dataset, self).__init__()

        self.data_smile = data_smile
        self.data_x = data_x

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):

        mol_feat = self.data_x[index]
        mol_smile = self.data_smile[index]


        mol_feat = np.array(mol_feat, dtype=np.float32)
        return mol_smile, mol_feat



def run_test(loader, model):
    model.eval()

    smile_list = []
    y_pred = []
    
    with torch.no_grad():

        for batch_i, (mol_smile, mol_feat) in enumerate(loader):
            print(f"batch i: {batch_i}/{len(loader)}")
            mol_feat = mol_feat.to(DEVICE)

            binary_logits = model(mol_feat)
            binary_sigmoid = nn.Sigmoid()(binary_logits)

            y_pred.extend(binary_sigmoid.detach().cpu().tolist())
            smile_list.extend(mol_smile)


    return smile_list, y_pred

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default='example.pkl', type=str)
    parser.add_argument('-o', '--output_path', default='example.csv', type=str)
    parser.add_argument('--aug', default="no_aug", choices= ["aug", "no_aug"], type=str)
    parser.add_argument('--trained_dataset', default="bindingdb", choices= ["bindingdb", "fda", "smm", "G4", "Hairpin", "PK", "TH", "TWJ", "multi_label"], type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_worker', default=4, type=int)

    parser.add_argument('--device', default="gpu", type=str)

    args = parser.parse_args()

    print(args)


    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("gpu is not available, run on cpu")
            device = torch.device("cpu")
 
    if args.aug == "aug":
        model_dir = "/Arontier_2/Projects/rna_binder/230804/aug"
    elif args.aug == "no_aug":
        model_dir = "/Arontier_2/Projects/rna_binder/230804/no_aug"
    else:
        print("Wrong aug argument input!!!!")
        exit(0)
    
    if args.trained_dataset == "bindingdb":
        model_subdir = "bindingdb"
    elif args.trained_dataset == "fda":
        model_subdir = "fda"
    elif args.trained_dataset == "smm":
        model_subdir = "smm"
    elif args.trained_dataset == "G4":
        model_subdir = "binary_G4"
    elif args.trained_dataset == "Hairpin":
        model_subdir = "binary_Hairpin"
    elif args.trained_dataset == "PK":
        model_subdir = "binary_PK"
    elif args.trained_dataset == "TH":
        model_subdir = "binary_TH"
    elif args.trained_dataset == "TWJ":
        model_subdir = "binary_TWJ"
    elif args.trained_dataset == "multi_label":
        model_subdir = "multi_label"
    else:
        print("Wrong trained_dataset argument input!!!!")
        exit(0)



    ### get unimol features
    feat = pd.read_pickle(args.input_path)
    smi_list, repr_list = [], []
    for batch in feat:
        sz = batch["bsz"]
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            repr_list.append(batch["mol_repr_cls"][i])


    smi_list = np.array(smi_list)
    repr_list = np.array(repr_list)

    test_dataset = rna_bind_dataset(smi_list, repr_list)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, pin_memory=True,
        collate_fn = collate_fn,
        num_workers=args.num_worker)
    
    if args.trained_dataset == "multi_label":
        class_num = 5
    else:
        class_num = 1
    model = rna_bind_mlp(class_num=class_num)
    model = model.to(device)

    model_dir = f"{model_dir}/{model_subdir}"
    ensemble_model_paths = glob.glob(f'{model_dir}/**/*.pth', recursive=True)

    y_pred_list = []
    for fold_i, model_path in enumerate(ensemble_model_paths):
        print(f"ensemble fold i: {fold_i}/{len(ensemble_model_paths)}")
        ch = torch.load(model_path)
        model.load_state_dict(ch['conv_model'])
        model.to(device)

        smile_list, y_pred = run_test(test_loader, model)
        y_pred_list.append(y_pred)

    y_pred_np_list = np.array(y_pred_list).squeeze()
    y_pred_avg = np.mean(y_pred_np_list, axis=0)

    if args.trained_dataset == "multi_label":
        df = {
            "smiles":smile_list,
            "G4":y_pred_avg[:,0],
            "Hairpin":y_pred_avg[:,1],
            "PK":y_pred_avg[:,2],
            "TWJ":y_pred_avg[:,3],
            "TH":y_pred_avg[:,4],
        }
    else:
        df = {
        "smiles":smile_list,
        "pred_proba":y_pred_avg,
        }


    df = pd.DataFrame.from_dict(df)
    df.to_csv(args.output_path, index=False, header=True)












