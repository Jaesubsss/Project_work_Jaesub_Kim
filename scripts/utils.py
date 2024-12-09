from torch.utils.data import Dataset
from torch import Tensor
import numpy as np



class OmicsDataset(Dataset): # PyTorch의 dataset class를 상속받아서 새로운 클래스를 정의
    def __init__(self, omic_dict, drug_dict, data): # cell linse에 대한 딕셔너리, 약물 딕셔너리, 데이터를 받아서 저장 
        self.omic_dict = omic_dict
        self.drug_dict = drug_dict
        self.cell_mapped_ids = {key:i for i, key in enumerate(self.omic_dict.keys())}
        # omic_dict의 키를 고유한 인덱스로 매핑
        # enumerate는 키들을 순서대로 열거하여 (인덱스, 키) 형태의 튜플로 반환
        # 딕셔너리 컴프레헨션: 각 키를 key로, 각 키의 인덱스를 i로 사용하여 {key:i}형태로 매핑된 딕셔너리 만듬.
        self.drug_mapped_ids = {key:i for i, key in enumerate(self.drug_dict.keys())}
        self.data = data
        
    def __len__(self):
        return len(self.data)
        # 데이터셋 크기
        
    def __getitem__(self, idx): # 인덱스를 받아, 인덱스에 해당하는 샘플을 반환
        instance = self.data.iloc[idx] # 해당 인덱스의 위치에 있는 데이터 (cell ID, drug ID, target value)
        cell_id = instance.iloc[0]
        drug_id = instance.iloc[1]
        target = instance.iloc[2]
        return (self.omic_dict[cell_id],
                self.drug_dict[drug_id],
                Tensor([target]),
                Tensor([self.cell_mapped_ids[cell_id]]),
                Tensor([self.drug_mapped_ids[drug_id]])) 
    
    
import rdkit # 분자구조와 화합물에 대한 정보를 다루는 라이브러리
from rdkit.Chem import AllChem
class FingerprintFeaturizer():
    # 화합물의 SMILES 문자열로부터 fingerprint를 추출하여 feature vector로 변환함.
    def __init__(self,
                 fingerprint = "morgan", # morgan fingerprint 유형을 사용한다. 주로 원자 주변의 원자 구조를 반영하는 방식이며, 원의 반경 R을 지정할 수 있다. 
                 R=2, 
                 fp_kwargs = {},
                 transform = Tensor):
        """
        Get a fingerprint from a list of molecules.
        Available fingerprints: MACCS, morgan, topological_torsion
        R is only used for morgan fingerprint.
        fp_kwards passes the arguments to the rdkit fingerprint functions:
        GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint, GetTopologicalTorsionFingerprint
        """
        self.R = R
        self.fp_kwargs = fp_kwargs # fingerprint에 추가적으로 전달할 인자들
        self.fingerprint = fingerprint
        if fingerprint == "morgan":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, self.R, **fp_kwargs)
            # 파이썬에서 **는 포인터가 아닌 keyword argument를 전달할 때 사용된다. **는 딕셔너리를 함수의 인수로 전달할 때 그 내용을 키-값 쌍으로 분해하여 전달한다. 
        elif fingerprint == "MACCS":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(x, **fp_kwargs)
        elif fingerprint == "topological_torsion":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(x, **fp_kwargs)
        self.transform = transform
        
    def __call__(self, smiles_list, drugs = None):
        # SMILES 문자열 리스트를 받아, 각 문자열을 RDKit의 Mol 객체로 변환하고, 변환된 분자를 self.f 함수에 전달하여 fingerprint 벡터로 변환한다.
        # 이 때, 변환에 실패하면 None을 반환해 문제 발생 시 유연하게 처리한다. 
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
            # 기본값(None)일 때, 순서대로 인덱스를 부여한다.
            
        for i in range(len(smiles_list)):
            try:
                smiles = smiles_list[i]
                molecule = AllChem.MolFromSmiles(smiles) # SMILES 문자열을 RDKit의 Mol 객체로 변환
                feature_list = self.f(molecule)
                f = np.array(feature_list)
                if self.transform is not None:
                    f = self.transform(f)
                drug_dict[drugs[i]] = f
            except:
                drug_dict[drugs[i]] = None
        return drug_dict
    
    def __str__(self):
        """
        returns a description of the featurization
        """
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"