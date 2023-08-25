# RNA_Binder
Predict RNA binder using Uni-Mol representation from smiles

## Requirements

python=3.8  
pytorch=2.0.0 (with cuda=11.8)  
unicore=0.0.1 (download wheel from https://github.com/dptech-corp/Uni-Core/releases)  
rdkit=2021.09.5  
```
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl
```

## Trained Weights Location in Arontier Server
### Pre-trained Uni-Mol Weight
```/Arontier_2/Projects/rna_binder/230804/pretrained_unimol```
### Trained Weights using ROBIN Data without Augmentation
```/Arontier_2/Projects/rna_binder/230804/no_aug```
### Trained Weights using ROBIN Data with Augmentation
```/Arontier_2/Projects/rna_binder/230804/aug```



## Run

### (1) Smiles to 3D Conformations



### (2) 3D Conformations to Unimol Representation



### (3) Unimol Representation to Predict Probability being a RNA binder 








## Reference

"Predicting Ligand – RNA Binding Using E3-Equivariant Network and Pretraining"  
"Machine Learning Informs RNA-Binding Chemical Space"  

https://github.com/dptech-corp/Uni-Core  
https://github.com/dptech-corp/Uni-Mol  
https://github.com/ky66/ROBIN  


