# Outlier Interpretation 

Note that this task is also referred to as outlier explanation, outlier aspect mining/discovering, outlier property detection, and outlier description.


### Eleven Outlier Interpretation Methods.

**This repository contains eleven outlier interpretation methods: TMQ[1], ATON [2], COIN[3], SiNNE[4], SHAP[5], LIME[6], Anchors [7], RAE-GA [8], PARs
[9], PML[10], and WANDER [11].**

[1] Optimizing latent space based on Transformer-guided quadruplet network for outlier interpretation. 

[2] Beyond outlier detection: outlier interpretation by attention-guided triplet deviation network. In WWW. 2021.

[3] Contextual outlier interpretation. In IJCAI. 2018.

[4] A new dimensionality-unbiased score For efficient and effective outlying aspect mining. In Data Science and Engineering. 2022.

[5] A unified approach to interpreting model predictions. In NeuraIPS. 2017.

[6] "Why should I trust you?" Explaining the predictions of any classifier. In SIGKDD. 2016.

[7] Anchors: high precision model-agnostic explanations. In AAAI. 2018.

[8] Outlier interpretation using regularized auto encoders and genetic algorithm. In CEC, 2024.

[9] PARs: predicate-based association rules for efficient and accurate anomaly explanation. In CIKM, 2024.

[10] Perspective-based multi-task learning for outlier interpretation. In Dassfa, 2025.

[11] Weighted-digraphguided multi-kernelized learning for outlier explanation. In Information Fusion, 2025.

  

### 1. Environment Setup & Dependency Installation
To replicate our experimental environment, please install the required packages. We recommend using a Python virtual environment. 
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
pyod==0.8.2
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
alibi==0.5.5
```
You can install them via pip using the command:
```
pip install torch==1.3.0 numpy==1.15.0 pandas==0.25.2 scikit-learn==0.23.1 pyod==0.8.2 tqdm==4.48.2 prettytable==0.7.2 shap==0.35.0 lime==0.2.0.1 alibi==0.5.5
```

### 2. Data Preparation
All datasets used in our experiments are located in the data/ folder. No additional preprocessing is required before running the main scripts.

### 3. Reproducing Experiments
The code supports multiple explanation algorithms. Follow the specific instructions for each group of methods.

**3.1 For TMQ' and Competitors: COIN, SHAP, LIME, and ATON**
(1) Open the main.py file.

(2) Locate and modify the algorithm_name variable to one of the following supported values (in lowercase): tmq, aton, coin, shap, lime.

(3) Run the experiment from the command line:
      ```
      python main.py --path data/ --runs 10
      ```
(4) Output: Results will be saved in the record/[algorithm_name]/ directory.

**3.2 For TMQ and Competitors: ATON' and COIN'**
1. In main.py, set the algorithm_name variable to tmq, aton, or coin.
2. Use the appropriate command based on the algorithm:
   For ATON' and TMQ:
   ```
   python main.py --path data/ --w2s_ratio auto --runs 10
   ```
   For COIN':
   ```
   python main.py --path data/ --w2s_ratio pn --runs 10
   ```
   The --w2s_ratio parameter controls the weighting strategy.
   
**3.3 For Competitors: SiNNE and Anchors**
1. Open the main2.py file.
2. Set the algorithm_name variable to sinne or anchor.
3. Execute the following command:
   ```
   python main2.py --path data/ --runs 10
   ```

### 4. Key Hyperparameters & Configuration
algorithm_name: The primary switch to select the explanation algorithm.

--path: Path to the dataset folder.

--runs: Number of experimental repetitions for robust result averaging.

--w2s_ratio: A key parameter for TMQ, ATON', and COIN' (auto for TMQ, pn for ATON'/COIN').

Most other hyperparameters have been set to the optimal values used in our paper and are configurable within the respective script files.

### 5. Ground-Truth Annotations for Evaluation

Please also find the Ground-truth outlier interpretation annotations in folder `data_od_evaluation`.   
*We expect these annotations can foster further possible reasearchs on this new practical probelm.*  

You may find that each dataset has three annotation files, please refer to the detailed annotation generation process in our submission.
**How to generate the ground-truth annotations:**
>  We employ eight different kinds of representative outlier detection methods (i.e., ensemble-based method iForest, probability-based method COPOD and ECOD, and proximity-based method HBOS and ROD, neural networks-based method SO_GAAL, graph-based method LUNAR, linear-based method MCD) to evaluate outlying degree of real outliers given every possible subspace. A good explanation for an outlier should be a high-contrast subspace that the outlier explicitly demonstrates its outlierness, and outlier detectors can easily and certainly predict it as an outlier in this subspace. Therefore, the ground-truth interpretation for each outlier is defined as the subspace that the outlier obtains the highest outlier score among all the possible subspaces.

### Loss curves for 12 real-world datasets
<img width="161" height="135" alt="losses_glass" src="https://github.com/user-attachments/assets/f62947f0-6e99-4b76-a704-1567b6e7b5e0" />
<img width="161" height="161" alt="losses_ann" src="https://github.com/user-attachments/assets/7952a21e-1a62-49f2-841a-72c732b3eaef" />
<img width="161" height="161" alt="losses_fault" src="https://github.com/user-attachments/assets/d3a72c33-0ddd-4a51-8e5e-2f2226129ead" />
<img width="161" height="161" alt="losses_fraud" src="https://github.com/user-attachments/assets/2e3ea768-9220-4ce1-83a3-889f5f3eb967" />
<img width="161" height="161" alt="losses_mamm" src="https://github.com/user-attachments/assets/1fa75765-a834-4470-9a81-b57a8c9f8af3" />
<img width="161" height="161" alt="losses_mnist" src="https://github.com/user-attachments/assets/16f0ffa7-33da-4567-9169-06ebb37b4630" />
<img width="161" height="161" alt="losses_page" src="https://github.com/user-attachments/assets/378c490b-c8c3-4bb4-848e-ff99879c8187" />
<img width="161" height="161" alt="losses_satimage" src="https://github.com/user-attachments/assets/0af3a2cc-eb4c-4cf5-b38c-6525b9437f1b" />
<img width="161" height="161" alt="losses_smtp" src="https://github.com/user-attachments/assets/870d5114-0767-403a-8951-50c452c1b817" />
<img width="161" height="161" alt="losses_vertebral" src="https://github.com/user-attachments/assets/10efe3ea-5652-4b92-8653-60ab854b6639" />
<img width="161" height="161" alt="losses_wpbc" src="https://github.com/user-attachments/assets/1da615c1-647a-42e0-a4c7-a3de3ac8d9d7" />
<img width="161" height="161" alt="losses_yeast" src="https://github.com/user-attachments/assets/0532cbeb-04b1-4ecb-ac2e-0a130208f822" />


### Convergence Analysis
We introduce an important Theorem to prove the convergence of TMQ.
<img width="656" height="784" alt="d061d94c91a33700f790b81dda552d36" src="https://github.com/user-attachments/assets/9544d955-f759-40e2-8566-874092995156" />
<img width="656" height="576" alt="c75d6dad3d09437a1c80a898812f9a7d" src="https://github.com/user-attachments/assets/e612493a-9cfc-4ff5-942d-713030b1365c" />

### JD's Dataset
<img width="2376" height="134" alt="image" src="https://github.com/user-attachments/assets/f0283b7a-7a71-4080-ba89-26a7a43f8681" />

<img width="2350" height="250" alt="image" src="https://github.com/user-attachments/assets/c086361d-8c3a-4d10-8ca6-083a34643f48" />



