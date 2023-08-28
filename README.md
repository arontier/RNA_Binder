# RNA_Binder
Predict RNA binder using Uni-Mol representation from smiles

## Requirements

python=3.8  
pytorch=2.0.0 (with cuda=11.8)  
unicore=0.0.1 (download wheel from https://github.com/dptech-corp/Uni-Core/releases)  
rdkit=2021.09.5  
pandas=2.0.3
lmdb=1.4.1
scikit-learn=1.3.0
scipy=1.8.0
numpy=1.22.3
```
### commends for install pytorch and unicore
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
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


```
CUDA_VISIBLE_DEVICES=0 python Uni-Mol/unimol/unimol/infer.py example.lmdb --results-path example.pkl \ 
--user-dir Uni-Mol/unimol/unimol --path /Arontier_2/Projects/rna_binder/230804/pretrained_unimol/mol_pre_no_h_220816.pt \
--num-workers 8 --ddp-backend=c10d --batch-size 32 --task unimol --loss unimol_infer --arch unimol_base \ 
--fp16 --fp16-init-scale 4 --fp16-scale-window 256 --only-polar 0 --dict-name dict.txt --log-interval 50 \ 
--log-format simple --random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer
```


### (3) Unimol Representation to Predict Probability being a RNA binder 








## Reference

"Predicting Ligand – RNA Binding Using E3-Equivariant Network and Pretraining"  
"Machine Learning Informs RNA-Binding Chemical Space"  

https://github.com/dptech-corp/Uni-Core  
https://github.com/dptech-corp/Uni-Mol  
https://github.com/ky66/ROBIN  


