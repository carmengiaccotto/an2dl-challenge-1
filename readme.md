# Multivariate Time Series Classification for Pain Assessment
# Artificial Neural Networks and Deep Learning - First Challenge
![](an2dl-challenge-1/blob/master/images/logo.png)

*Artificial Neural Network and Deep Learning* is a course offered by [Politecnico di Milano](https://www.polimi.it).
In this repository you can find the Jupyter Notebooks that we created for the 2025 AN2DL Firts Challenge on [kaggle](https://www.kaggle.com/competitions/an2dl2526c1/overview).
You can read our final report [here](https://github.com/carmengiaccotto/an2dl-challenge-1/blob/master/AN2DL25__Challenge1.pdf).

## Dataset: Pirate Pain Dataset
The provided dataset consists of **multivariate time series data**, with each patient having a sequence of **160 time steps**.

### Key Characteristics
The dataset includes:
* **Continuous Features (Joint Angles):** 31 features related to joint angles (e.g., `joint_00` - `joint_30`) measured over time.
* **Static Attributes (Patient Characteristics):** categorical attributes describing patient characteristics (e.g., `n_legs`, `n_hands`, `n_eyes`) and pain survey results (`pain_survey_1` - `pain_survey_4`).

## Task
Predict the true pain level of each subject (**no_pain**, **low_pain**, **high_pain**) based on their time-series motion data.

## Main Challenge: Class Imbalance
A primary challenge was the **severe class disparity** across the three labels. The following strategies were adopted to mitigate this risk:
- **Stratified Cross-Validation:** to maintain consistent class ratios in each split.
- **Weighted Cross-Entropy Loss:** by assigning higher penalty weights to the rare classes (**low pain** and **high pain**) to ensure the model prioritized learning the clinically critical minority samples.

## The Model: Bidirectional LSTM
The best-performing final model is a **Bidirectional Long Short-Term Memory (LSTM)**, optimized for processing temporal sequences.

### Optimal Final Configuration
- **Architecture:** Bidirectional LSTM.
- **HIDDEN LAYERS:** 2.
- **HIDDEN SIZE:** 128.
- **Optimizer:** AdamW from PyTorch.
- **Loss Function:** CrossEntropyLoss with manually rescaled class weights and label smoothing.
- **Additional Techniques:** Gradient Clipping was incorporated.

### Pre-processing and Feature Engineering
- **Initial Cleaning:** removal of constant features and highly correlated variables to reduce redundancy, based on the **Correlation Matrix**. 
- **Categorical Features:** string values (`n_legs`, `n_hands`, `n_eyes`) were mapped to integer indices and processed through custom embedding vectors to allow the model to autonomously learn meaningful representations.
- **Temporal Normalization:** the `timestamp` column was normalized to the range [0, 1].
- **Data Normalization:** application of the scikit-learn **StandardScaler** object, fitted only on the train set to prevent data leakage.

### Explored and Discarded Techniques
Several strategies did not yield the expected improvements:
- **Principal Component Analysis (PCA)**
- **1D Convolutional Layers**
- **Attention Layer** 
- **Outlier Filtering**

## Results
- Public Score: **0.94044**
- Private Score: **0.94532**

## Team: ANeuronInFour
- *[Carmen Giaccotto](https://github.com/carmengiaccotto)*
- *[Simone Pio Bottaro](https://github.com/SimonePioBottaro)*
- *[Davide Bertoni](https://github.com/Bert0ns)*
- *[Francesco Lauria](https://github.com/FRANCESC0LAUR1A)*
