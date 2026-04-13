# A5 Toxicity Classification

## Install & Run
```sh
pip install scikit-learn xgboost pandas numpy matplotlib seaborn tqdm
python3 main.py
```

## Pipeline phases
| Phase | Description |
|-------|-------------|
| 1 | Data loading + EDA → 6 plots saved to `./plots/` |
| 2 | Preprocessing (variance filter, imputation, scaler) |
| 3 | Baseline models (LR, RF) with fold-by-fold BCR progress |
| 4 | Hyperparameter tuning (XGBoost / SVM / MLP + voting ensemble) |
| 5 | Feature selection (SelectKBest, mutual info, PCA) + plot |
| 6 | Final OOF BCR evaluation, confusion matrix, fold BCR chart |
| 7 | Predictions → `predictions.csv` |

## Plots generated (`./plots/`)
`eda_01_class_distribution` · `eda_02_feature_variances` · `eda_03_correlation_heatmap`
`eda_04_distribution_shift` · `eda_05_outlier_features` · `eda_06_top_discriminative_features`
`model_01_comparison` · `model_02_xgb_feature_importance` · `model_03/04/05_search`
`feat_01_selection_comparison` · `eval_01_confusion_matrix` · `eval_02_fold_bcr_scores`

---

# This is a README
A5 - A Machine Learning Competition

This final (and longer) project will give you the opportunity to have some fun while competing with other students to solve a real and difficult prediction task inspired by a biomedical research topic. Your job will be twofold:
* build the best possible classification model on a training set to predict the (undisclosed) class labels on a given test set,
* predict the actual classification performance your model will have on the test set.

Design choices
The "training" set is referred here in a broad sense, that is the fraction of the dataset on which the class labels are disclosed. It is up to you to decide what to do with this labeled set and, for instance, whether you split it (once or several times, possibly recursively) into actual training versus some validation fraction. More generally, this project is intended to be open as it is the case for a real task. Only the outcome matters! Many design choices are left open and you must specify them. Here is a non-exhaustive list of things you might have to consider.
* Do you need to pre-process, to filter out, to normalize, etc., the available data? (look at the data,... LOOK AT THE DATA!)
* How are you going to address the fact that some feature values could be numerical (either int or float) while others could be categorical or possibly boolean? Are there specific feature values only observed in the test but not in the training?
* Should you consider all the available features? Should you define new or additional features from the existing ones? Should the newly defined features be crafted by hand, or generated automatically by other methods?
* Which methodology are you going to use to learn a model, to fix its possible hyper-parameters and to predict its classification performance on the test set?
* Which learning algorithm do you consider? You are free to choose the one you estimate more appropriate to the task. It needs not be (but it might be, of course) a learning algorithm that has been presented in the course. You can even consider several learning algorithms and/or produce and combine several models as long as your final prediction defines a unique label for each test example (ensemble methods? bagging?...).
* What else?... The sky (and computing power) is the limit!

Competition performance metric
Since the dataset needs not be perfectly balanced in terms of class priors, the chosen test performance metric of a classifier is defined as the balanced classification rate (BCR). The BCR computes the classification rate for each class and reports the arithmetic average of those rates over all classes. In a binary classification context, BCR is simply the average between specificity and sensitivity. More generally, with
C
classes:
BCR=
1
C

(
∑
i=1
C
T
P
i
n
i

)
where:
* nidenotes the number of examples from class iincluded in the test set,
* TPidenotes the number of correctly classified examples from class iin the test set,
* Cdenotes the number of classes.
Note that
T
P
i
n
i

is also denoted below as
p
i
and simply refers to the proportion of correctly classified test examples from class
i
.
Note also that a trivial classifier predicting all examples as belonging to the same class has a test
BCR=
1
C

, no matter how the class priors are distributed (whenever the single predicted class is never actually observed in the test set, BCR = 0 assuming
0
0

= 0.), while a perfect classifier has
BCR=1
(what is the expected BCR of a classifier predicting uniformly at random among the classes? Does it depend on the class priors?).
Since your task is not only to produce a model with the best possible BCR on the test set but also to predict how well your model is going to perform on this test set, we distinguish between the true test
BCR
of your model and your predicted test
BCR
ˆ
on this test set. The actual competition performance metric
P
(according to which you will be ranked and graded) is as follows:
P=BCR−Δ(BCR)⋅[1−exp(
−Δ(BCR)
σ

 )]
where:
* Δ(BCR)=|BCR−BCRˆ|

* σ=1C∑Ci=1pi(1−pi)ni‾‾‾‾‾‾‾‾‾‾‾√, where piis the proportion of correctly classified test examples from class iand niis the number of test examples from class i.
This competition performance metric is directly inspired from the International Performance Prediction Challenge WCCI 2006. The larger
P
the better. To maximize
P
you need to get the best possible test
BCR
and to make sure that your predicted
BCR
ˆ
is as close as possible to the actual test
BCR
of your model. Any deviation between both is penalized (by
−Δ(BCR)
) while the influence of such penalty is limited in the region of uncertainty where
Δ(BCR)
is commensurate with
σ
, the error bar on your true test BCR.
Here is a typical plot of the performance metric
P
as a function of the predicted
BCR
ˆ
for a true test
BCR
being equal to 70 %.
￼
The only thing you will need to provide for us to compute
P
is your predicted
BCR
ˆ
and the predicted class labels of the test examples. From this and knowing the undisclosed true class labels, we will compute the actual
BCR
of your model, its estimated error bar
σ
and the resulting
P
. The highest
P
among all submissions will define the winner of the competition (and earn our congratulations!).
Note that you will not know your
P
score before the closing of the competition, not even your rank among the other competitors. This is the game!

The prediction task

You are a data analyst and machine learning expert in a chemical lab and you have been tasked to study the toxicity of different chemical products based on the molecules that compose them and how present they are.
More precisely, your task is to decide if a given product is toxic (positive) or not toxic (negative).
￼
Generated with ChatGPT on 11th February 2026.

You are provided with a partially annotated dataset. The train split contains 3,000 examples and the test split contains 1,000 examples. Each feature corresponds to the quantity of the corresponding molecule in the product. They are obtained through a long, tidious, noisy, and difficult process (that is not relevant for this project).
Each data split includes 1,024 input float features.

The output class to be predicted:
* The label variable (categorical) indicates the toxicity of the product: positive, negative.


The data, available on Moodle, is split into 3 files:
* A5_2026_train.csv which contains the 1,024 input features of 3,000 examples to train and validate your models.
* A5_2026_train_labels.csv which contains the corresponding classes of the 3,000 training (and validation) examples and their index.
* A5_2026_test.csv which contains the 1,024 input features of 1000 test examples. It is your task to predict one class for each test example.
Once these files are stored on your local disk in the current working directory of a python session, the data can easily be loaded in pandas dataframes as follows.
import pandas as pd

x_train = pd.read_csv("A5_2026_train.csv")
y_train = pd.read_csv("A5_2026_train_labels.csv", index_col=0)
x_test = pd.read_csv("A5_2026_test.csv")



Question 1: Prediction

Upload a .csv file containing your predictions on the test set. The first line is a comma (for the index column) and the name of column (i.e. label). Each following lines must contain the index (the corresponding line in the test set, [0, 999]) and the predicted class for the corresponding test example.
Here is an example of correct format:
,label
0,positive
1,positive
2,negative
3,negative
4,positive
...
999,negative
Assuming the predicted classes of the test examples are stored in a variable y_pred (that is, a pandas.Series with y_pred.name == 'label'), one can easily produce such a .csv file with the following code:
import pandas as pd

y_pred.to_csv('predictions.csv', index=True)

Taille fichier max. : 1.0 MiB
Extensions autorisées : .csv
Question 2: Expected BCR

Give your expected BCR for the test set (range between 0 and 1).
For example: 0.5

Question 3: Code

Please copy-paste the code you used for this competition.
It must contain at least:
* the complete code for training your final model, including how you pre-processed the data;
* the code for predicting the test classes with your final model;
* and how you have computed the expected test BCR.
Your code should execute without errors when downloaded and run locally (please use relative paths when loading your train or test data, e.g. ./A5_2026_test.csv).
Warning: unlike in previous assignments, Inginious does not execute your code at the time of submission, i.e. it will not yield any warning in case of syntactic or runtime errors.
Important note
Please add comments to your code to explain the different parts. You must also cite any public external resources
 
1
# Note that, to be evaluated in this project, you **must** replace this comment by your own code
2
# that you have used to produce the answers to the previous questions.
3
​


Question 4: Report

Upload a report explaining your design choices, which algorithms/software you have used to perform the classification, and the exact protocol you followed to evaluate your model(s) and to produce your final prediction on the test set. Add any information you deem relevant (e.g. dealing with large number of features, comparison between approaches, etc.). Note that you can also include plots (e.g. cross-validated classification results when fine-tuning meta-parameters) but then you should analyze and comment the plots to tell the reader which conclusions you draw from it. You should target a report of 2 or 3 pages (about 1000 - 2000 words) of informative content.
Warning
Failure to submit an informative report would imply a penalty of up to 20 % of this project grade.

Taille fichier max. : 11.4 MiB
Extensions autorisées : .pdf

You are an expert machine learning engineer and data scientist helping with a 
binary toxicity classification competition (LINFO2262 - A5).

TASK: Predict whether chemical products are toxic (positive) or not toxic 
(negative) based on 1,024 molecular quantity features.

DATASET:
- Train: 3,000 examples × 1,024 float features
- Test: 1,000 examples × 1,024 float features
- Files: A5_2026_train.csv, A5_2026_train_labels.csv, A5_2026_test.csv

COMPETITION METRIC (P):
P = BCR − Δ(BCR) · [1 − exp(−Δ(BCR)/σ)]
where BCR = balanced classification rate, BCRhat = your predicted BCR,
Δ(BCR) = |BCR − BCRhat|, and σ = error bar on BCR.
Maximizing P requires both a high BCR AND an accurate BCR prediction.

OUTPUT FORMAT (predictions.csv):
,label
0,positive
1,negative
...
999,negative

YOUR RESPONSIBILITIES:
- Write clean, executable Python code using relative paths
- Always reason about whether preprocessing steps are justified by the data
- Track BCR estimates rigorously using cross-validation
- Produce a final BCRhat that is honest and well-calibrated (not overconfident)
- Keep code modular so components (preprocessing, model, evaluation) are 
  easy to swap or compare