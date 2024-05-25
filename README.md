# Outlier Interpretation 

Note that this task is also referred to as outlier explanation, outlier aspect mining/discovering, outlier property detection, and outlier description.


### Seven Outlier Interpretation Methods (The source code of the proposed TMQ will be publicly available upon acceptance.)

**This repository contains seven outlier interpretation methods: TMQ[1], ATON [2], COIN[3], SiNNE[4], SHAP[5], LIME[6], and Anchors [7].**

[1] Optimizing Latent Space based on Transformer-Guided Quadruplet Network for Outlier Interpretation. 

[2] Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network. In WWW. 2021.

[3] Contextual outlier interpretation. In IJCAI. 2018.

[4] A New Dimensionality-Unbiased Score For Efficient and Effective Outlying Aspect Mining. In Data Science and Engineering. 2022.

[5] A unified approach to interpreting model predictions. In NeuraIPS. 2017

[6] "Why should I trust you?" Explaining the predictions of any classifier. In SIGKDD. 2016.

[7] Anchors: High Precision Model-Agnostic Explanations. In AAAI. 2018.



### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of TMQ and its contenders  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments



### How to use?
##### 1. For TMQ' and competitor COIN, SHAP, and LIME, and ATON
1. modify variant `algorithm_name` in `main.py` (support algorithm:`tmq`, `aton`, `coin`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs 10 `
3. the results can be found in `record/[algorithm_name]/` folder  

##### 2. For TMQ and competitor ATON', COIN' 
1. modify variant `algorithm_name` in `main.py` to `tmq`, `aton` or `coin`  
2. use `python main.py --path data/ --w2s_ratio auto --runs 10` to run TMQ  
   use `python main.py --path data/ --w2s_ratio pn --runs 10` to run ATON', COIN'  

##### 3. For competitor SiNNE and Anchors
1. modify variant `algorithm_name` in `main2.py` to `sinne` or `anchor`  
please run `python main2.py --path data/ --runs 10` 

  

### Requirements
main packages of this project  
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



### Ground-truth annotations

Please also find the Ground-truth outlier interpretation annotations in folder `data_od_evaluation`.   
*We expect these annotations can foster further possible reasearchs on this new practical probelm.*  

You may find that each dataset has three annotation files, please refer to the detailed annotation generation process in our submission.
**How to generate the ground-truth annotations:**
>  We employ eight different kinds of representative outlier detection methods (i.e., ensemble-based method iForest, probability-based method COPOD and ECOD, and proximity-based method HBOS and ROD, neural networks-based method SO_GAAL, graph-based method LUNAR, linear-based method MCD) to evaluate outlying degree of real outliers given every possible subspace. A good explanation for an outlier should be a high-contrast subspace that the outlier explicitly demonstrates its outlierness, and outlier detectors can easily and certainly predict it as an outlier in this subspace. Therefore, the ground-truth interpretation for each outlier is defined as the subspace that the outlier obtains the highest outlier score among all the possible subspaces.
