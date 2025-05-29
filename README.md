<center>
    <center>
        <img src = images/dataset-card.jpg width = 35%/>
    </center>
</center>

## Project Title

**Spotify Music Genre Classification Using Machine Learning Models**

**Author**
Xinhong Hu


## Abstract

This project leverages machine learning to classify music genres using Spotify audio features. Through preprocessing, visualization (PCA), and model building (Logistic Regression, Random Forest, CatBoost), it evaluates performance across multiple clusters and highlights key audio features for genre prediction. Results suggest CatBoost performs best for this multi-class problem, and the project provides a foundation for improving music recommendation systems.

## Dataset Information

This dataset contains Spotify music features for a music genre classification task. It is sourced from a publicly available Kaggle dataset, including various audio features and genre labels, suitable for machine learning classification.

## Business Understanding

### What is Music Genre Classification?

Music genre classification is a critical component of music recommendation systems and content analysis. Platforms like Spotify analyze audio features (e.g., tempo, loudness, energy) to categorize songs into genres (e.g., rock, jazz, pop), enhancing user experience, optimizing recommendation algorithms, and personalizing playlists. Traditional genre classification relies on manual labeling or simple rules, whereas machine learning enables automated, efficient classification.

### Project Objectives

- Explore the feature distribution of the Spotify music dataset.
- Visualize genre differences using Principal Component Analysis (PCA).
- Build multi-class classification models (Logistic Regression, Random Forest, CatBoost) to predict song genres.
- Evaluate model performance and analyze feature importance to inform music recommendation system improvements.

This is a supervised learning task, training models on labeled genre data to predict genres for new songs.

### Why Machine Learning?

Music genre classification involves high-dimensional features (e.g., acoustic properties, rhythm patterns) and complex categories (multiple genres). Machine learning models like Random Forest and gradient boosting capture nonlinear relationships, outperforming traditional rules. Logistic Regression offers interpretability, while advanced models like CatBoost excel in handling multi-class imbalanced data.

## Research Question

Can machine learning models accurately classify Spotify songs into genres based on their audio features?

## Methodology

1. **Preprocessing**: Clean, encode, and standardize audio feature data.
2. **Clustering**: Split dataset into 8 clusters based on feature similarity.
3. **Modeling**: Train and evaluate Logistic Regression, Random Forest, and CatBoost for each cluster.
4. **Visualization**: PCA for 2D genre separation; ROC and feature importance plots.
5. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, AUC.

### Data Sources

Kaggle public dataset, e.g., Spotify Tracks Dataset.

## Data Understanding

- Dataset Characteristics: Multivariate
- Domain: Music/Audio Analysis
- Attribute Characteristics: Real, Categorical
- Missing Values: Few (handled)

### Attribute Information

**Data File**: `train.csv`
Contains Spotify song audio features (e.g., loudness, tempo, energy) and genre labels (`track_genre`).
Format: CSV, with numerical features (real numbers) and categorical features (genre names).

**Feature Examples:**

* Numerical: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`
* Categorical: `track_genre` (e.g., pop, rock, jazz)

**Clustered Data**: Post-processed data split into 8 clusters, stored in `cluster_split_data/`, with training and test CSV files (`cluster_{id}_train.csv`, `cluster_{id}_test.csv`).

## Team Collaboration - Directory Structure

```plaintext
├── data
│    ├── train.csv
│    ├── cluster_split_data
│    │    ├── cluster_0_train.csv
│    │    ├── cluster_0_test.csv
│    │    ├── ...
├── models
│    ├── saved_models
│    │    ├── logistic
│    │    │    ├── cluster_0_model.joblib
│    │    │    ├── cluster_0_scaler.joblib
│    │    │    ├── ...
│    │    ├── random_forest
│    │    │    ├── cluster_0_model.joblib
│    │    │    ├── ...
│    │    ├── catboost
│    │    │    ├── cluster_0_model.cbm
│    │    │    ├── ...
│    │    ├── class_names
│    │    │    ├── logistic_cluster_0_classes.txt
│    │    │    ├── random_forest_cluster_0_classes.txt
│    │    │    ├── catboost_cluster_0_classes.txt
│    │    │    ├── ...
├── images
│    ├── pca_2d_visualization.png
│    ├── cluster_0_roc_comparison.png
│    ├── ...
├── coef_analysis_plots
│    ├── cluster_0_logistic_coef_class_pop.png
│    ├── ...
├── feature_importance_plots
│    ├── cluster_0_rf_feature_importance.png
│    ├── cluster_0_catboost_feature_importance.png
│    ├── ...
├── roc_plots
│    ├──cluster_0_roc_comparison
│    ├── ...
├── Spotify_pre_processing.ipynb
├── Spotify_pre_visualize.ipynb
├── Spotify_model_building_evaluating.ipynb
```
[Since some model files (random forest and catboost .joblib/.cbm files) were too large to upload to GitHub, I have deleted them.]
## Data Preparation and Visualization

### Code Used: Python

### Packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

### Preprocessing (`Spotify_pre_processing.ipynb`):

- Load `train.csv`, remove non-numeric columns (retain `track_genre`).
- Drop rows with missing values, encode `track_genre` using `LabelEncoder`.
- Standardize numeric features with `StandardScaler`.
- Split data by clusters, generating training and test sets, saved to `cluster_split_data/`.

### Visualization (`Spotify_pre_visualize.ipynb`):
- Explore the distribution of raw data using a variety of visualization methods.
- Apply PCA to reduce features to 2D, generating a scatter plot to show genre distribution.
- Output: `images/pca_2d_visualization.png`,...

## Data Cleansing, Processing, and Modeling

### Code Used: Python

### Packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `catboost`

### Modeling and Evaluation (`Spotify_model_building_evaluating.ipynb`):

**Data**: Use clustered data from `cluster_split_data/`.

**Models** (trained per cluster):

- **Logistic Regression** (`LogisticRegression`): Provides interpretable coefficients.
- **Random Forest** (`RandomForestClassifier`): Captures nonlinear relationships.
- **CatBoost** (`CatBoostClassifier`): Handles multi-class imbalanced data.

**Evaluation:**

- **Classification Report**: Precision, recall, F1-score, accuracy (e.g., Cluster 0 accuracy: 0.28, Cluster 1 accuracy: 0.36).
- **ROC Curves**: Compare AUC for all models (saved as `images/cluster_{id}_roc_comparison.png`).

**Feature Analysis:**

- **Logistic Regression**: Generate coefficient plots for feature impact per genre (`coef_analysis_plots/`).
- **Random Forest and CatBoost**: Generate feature importance plots (`feature_importance_plots/`).

**Outputs:**

- Model Files: `saved_models/{model}/cluster_{id}_model.{joblib,cbm}`
- Scalers: `saved_models/{model}/cluster_{id}_scaler.joblib`
- Class Names: `saved_models/class_names/{model}_cluster_{id}_classes.txt`

## Process Summary(result)

Through data preprocessing, visualization, and modeling, this project achieved:

- **Data Insights**: PCA visualization revealed genre differences in feature space.
- **Model Performance**: Trained and compared three models, with CatBoost showing strong performance in multi-class tasks.
- **Feature Analysis**: Identified key audio features (e.g., energy, tempo) for genre classification, informing recommendation system design.
- **Scalability**: Clustered data and modular code support large-scale datasets and additional models.

## Next Steps

- Introduce time-series or lyrics features.
- Explore model ensembling strategies.
- Support interactive 3D PCA visualizations.
- Integrate with a music recommendation engine.

## Conclusion

The project demonstrates that genre classification using machine learning is feasible and beneficial. CatBoost excels in handling complex, imbalanced datasets. Feature importance analysis reveals actionable insights for future feature engineering and system improvement.

## Bibliography

- Maharshi Pandya. “Spotify Tracks Dataset.” Kaggle, [https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Scikit-learn documentation
- CatBoost documentation
- Spotify Developer Guide















