import pandas as pd
import numpy as np
import torch
import scripts
from functools import lru_cache
import torchmetrics
from torch import nn
import optuna

@lru_cache(maxsize=None)
def get_data(n_fold = 0, fp_radius = 2, typ = "rnaseq"):
    # drug
    smile_dict = pd.read_csv("data/smiles.csv", index_col=0)
    fp = scripts.FingerprintFeaturizer(R = fp_radius)
    drug_dict = fp(smile_dict.iloc[:, 1], smile_dict.iloc[:, 0])
    
    # loading all datasets
    driver_genes = pd.read_csv("data/driver_genes.csv").loc[:, "symbol"].dropna()

    rnaseq = pd.read_csv("data/rnaseq_normcount.csv", index_col=0)
    driver_columns = rnaseq.columns.isin(driver_genes)
    filtered_rna = rnaseq.loc[:, driver_columns]
    
    proteomics = pd.read_csv("data/proteomics.csv", index_col=0)
    
    mutation = pd.read_csv("data/binary_mutations.csv")
    mutation.columns = mutation.iloc[0]
    mutation = mutation.iloc[2:,:].set_index("gene_symbol")
    driver_columns = mutation.columns.isin(driver_genes)
    filtered_mut = mutation.loc[:, driver_columns]
    filtered_mut = filtered_mut.astype(float)

    methylations = pd.read_csv("data/methylations.csv",index_col = 0).sort_index(ascending = True)

    cnvs = pd.read_csv("data/copy_number_variations.csv",index_col= 0)

    # concatenate all dataset 
    # inner join based on index: model_ids with NaN are automatically filtered out 
    data_concat = pd.concat([filtered_rna, proteomics, filtered_mut, methylations, cnvs], axis=1, join='inner')

    # choosing dataset to be used for training model
    if typ == "rnaseq":
        data_fin = filtered_rna[filtered_rna.index.isin(data_concat.index)]
    if typ == "proteomics":
        data_fin = proteomics[proteomics.index.isin(data_concat.index)]
    if typ == "mutations":
        data_fin = filtered_mut[filtered_mut.index.isin(data_concat.index)]
    if typ == "methylations":
        data_fin = methylations[methylations.index.isin(data_concat.index)]
    if typ == "cnvs":
        data_fin = cnvs[cnvs.index.isin(data_concat.index)]
    if typ == "concat":
        data_fin = data_concat
    
    tensor_concat = torch.Tensor(data_fin.to_numpy())
    cell_dict = {cell: tensor_concat[i] for i, cell in enumerate(data_fin.index.to_numpy())}
    
    # GDSC
    data = pd.read_csv("data/GDSC1.csv", index_col=0)
    # default, remove data where lines or drugs are missing:
    data = data.query("SANGER_MODEL_ID in @cell_dict.keys() & DRUG_ID in @drug_dict.keys()")
    unique_cell_lines = data.loc[:, "SANGER_MODEL_ID"].unique()
    
    np.random.seed(420) # for comparibility, don't change it!
    np.random.shuffle(unique_cell_lines)
    folds = np.array_split(unique_cell_lines, 10)
    test_lines = folds[0] # ?? 질문 필요. 0번째 fold를 test로 쓴다면, 이후에도 train_idx에서 0을 제거해야하는것 아닌가? 만약 train_idx에서 n_fold를 제거할 것이라면, trest_lines를 folds[n_fold]로 바꿔야하지 않는가? 
    train_idxs = list(range(10))
    train_idxs.remove(n_fold)
    np.random.seed(420)
    validation_idx = np.random.choice(train_idxs)
    train_idxs.remove(validation_idx)
    train_lines = np.concatenate([folds[idx] for idx in train_idxs])
    validation_lines = folds[validation_idx]
    test_lines = folds[n_fold] # 아 여기서 오버라이드 됐네? 그럼 위에 test_lines는 왜 있는거지?
    # 5
    train_data = data.query("SANGER_MODEL_ID in @train_lines")
    validation_data = data.query("SANGER_MODEL_ID in @validation_lines")
    test_data = data.query("SANGER_MODEL_ID in @test_lines")
    
    return (scripts.OmicsDataset(cell_dict, drug_dict, train_data),
    scripts.OmicsDataset(cell_dict, drug_dict, validation_data),
    scripts.OmicsDataset(cell_dict, drug_dict, test_data))