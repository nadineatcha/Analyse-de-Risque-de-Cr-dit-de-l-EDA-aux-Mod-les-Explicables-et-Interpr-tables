import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline;

# Chargement des données
df = pd.read_csv("german_credit_data.csv", index_col=0)

# Afficher les premières lignes
print(df.head())

# Informations sur le dataset
print(df.info())

# Statistiques descriptives
print(df.describe())

# Vérification des valeurs manquantes
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Nombre': missing_values, 'Pourcentage': missing_percent})
print(missing_df[missing_df['Nombre'] > 0])

# Distribution des variables numériques
numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    #plt.title(f'Distribution de {col}')
    #plt.show()

# Matrice de corrélation
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title('Matrice de Corrélation')
#plt.show()


# Distribution des variables catégorielles
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution de {col}')
    plt.xticks(rotation=45)
    plt.show()

# Si la colonne Risk existe
if 'Risk' in df.columns:
    # Relation entre variables numériques et Risk
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Risk', y=col, data=df)
        plt.title(f'{col} par niveau de risque')
        plt.show()
    
    # Relation entre variables catégorielles et Risk
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        cross_tab = pd.crosstab(df[col], df['Risk'], normalize='index')
        cross_tab.plot(kind='bar', stacked=True)
        plt.title(f'Risk par {col}')
        plt.xticks(rotation=45)
        plt.show()
  
  # Pour les variables catégorielles, créer une catégorie "Unknown"
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col].fillna("Unknown", inplace=True)

# Pour les variables numériques, utiliser la médiane
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

    # Conversion en variable binaire (0 pour "good", 1 pour "bad")
if 'Risk' in df.columns:
    df['Risk_binary'] = df['Risk'].map({'good': 0, 'bad': 1})

    from sklearn.model_selection import train_test_split

# Préparation des données pour le modèle
X = df.drop(columns=['Risk_binary', 'Risk'])
y = df['Risk_binary']

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identification des colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Création d'un pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Application du prétraitement
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Vérification de la forme des données après prétraitement
print(f"Forme de X_train après prétraitement: {X_train_preprocessed.shape}")
print(f"Forme de X_test après prétraitement: {X_test_preprocessed.shape}")

# Obtention des noms des caractéristiques après encodage
cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_feature_names])


# Application du prétraitement
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Pour statsmodels, conversion en tableau numpy si format sparse
X_train_array = X_train_preprocessed.toarray() if hasattr(X_train_preprocessed, 'toarray') else X_train_preprocessed
X_test_array = X_test_preprocessed.toarray() if hasattr(X_test_preprocessed, 'toarray') else X_test_preprocessed

# Ajout d'une constante pour l'intercept (nécessaire pour statsmodels)
X_train_sm = sm.add_constant(X_train_array)
X_test_sm = sm.add_constant(X_test_array)

# Récupération des noms des caractéristiques pour l'interprétation ultérieure
cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_feature_names])
feature_names_with_const = np.concatenate([['const'], feature_names])

import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

# Construction et entraînement du modèle
glm_model = sm.GLM(y_train, X_train_sm, family=Binomial())
glm_results = glm_model.fit()

# Affichage du résumé du modèle
print(glm_results.summary())

# Prédictions de probabilité
y_pred_proba_sm = glm_results.predict(X_test_sm)

# Conversion en prédictions binaires (0/1)
y_pred_sm = (y_pred_proba_sm >= 0.5).astype(int)


from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, f1_score, precision_recall_curve, average_precision_score
)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_sm)
print("Matrice de confusion:")
print(cm)

# Rapport de classification (précision, rappel, F1-score)
print("Rapport de classification:")
print(classification_report(y_test, y_pred_sm))

# F1-score (mesure équilibrée entre précision et rappel)
f1 = f1_score(y_test, y_pred_sm)
print(f"F1-score: {f1:.4f}")

# Courbe ROC et AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_sm)
roc_auc = auc(fpr, tpr)
print(f"AUC (Area Under ROC Curve): {roc_auc:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Ligne diagonale de référence
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (1 - Spécificité)')
plt.ylabel('Taux de vrais positifs (Sensibilité)')
plt.title('Courbe ROC du modèle GLM')
plt.legend(loc="lower right")
plt.savefig('roc_curve_glm.png')

# Création d'un DataFrame pour faciliter l'analyse des coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names_with_const[:len(glm_results.params)],
    'Coefficient': glm_results.params,
    'Std Error': glm_results.bse,
    'z-value': glm_results.tvalues,
    'P-Value': glm_results.pvalues,
    'Odds Ratio': np.exp(glm_results.params),
    'Significatif': glm_results.pvalues < 0.05
})

# Tri par importance (valeur absolue du coefficient)
coef_df_sorted = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
print("Coefficients triés par importance:")
print(coef_df_sorted)

plt.figure(figsize=(12, 10))
top_features = coef_df_sorted.iloc[:15]  # Top 15 caractéristiques
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.title('Coefficients les plus importants du modèle GLM')
plt.xlabel('Valeur du coefficient')
plt.ylabel('Caractéristique')
plt.axvline(x=0, color='r', linestyle='--')  # Ligne à zéro
plt.grid(axis='x')
plt.savefig('important_coefficients.png')
plt.figure(figsize=(12, 10))
plt.barh(top_features['Feature'], top_features['Odds Ratio'])
plt.title('Odds Ratios des caractéristiques les plus importantes')
plt.xlabel('Odds Ratio (échelle logarithmique)')
plt.ylabel('Caractéristique')
plt.xscale('log')  # Échelle logarithmique pour mieux visualiser
plt.axvline(x=1, color='r', linestyle='--')  # Ligne à OR=1 (pas d'effet)
plt.grid(axis='x')
plt.savefig('odds_ratios.png')

# Filtrer uniquement les caractéristiques statistiquement significatives
significant_features = coef_df[coef_df['P-Value'] < 0.05].sort_values(by='Coefficient', key=abs, ascending=False)
print("Caractéristiques statistiquement significatives (p < 0.05):")
print(significant_features)

# Filtrer uniquement les caractéristiques statistiquement significatives
significant_features = coef_df[coef_df['P-Value'] < 0.05].sort_values(by='Coefficient', key=abs, ascending=False)
print("Caractéristiques statistiquement significatives (p < 0.05):")
print(significant_features)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Création du pipeline avec l'arbre de décision
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))
])

# Entraînement du modèle
tree_pipeline.fit(X_train, y_train)

# Prédictions
y_pred_tree = tree_pipeline.predict(X_test)
y_pred_proba_tree = tree_pipeline.predict_proba(X_test)[:, 1]

# Évaluation
print("\nRapport de classification (Decision Tree):")
print(classification_report(y_test, y_pred_tree))

# F1-score
f1_tree = f1_score(y_test, y_pred_tree)
print(f"\nF1-score (Decision Tree): {f1_tree:.4f}")

# AUC
roc_auc_tree = roc_auc_score(y_test, y_pred_proba_tree)
print(f"AUC (Decision Tree): {roc_auc_tree:.4f}")

# Visualisation de l'arbre
plt.figure(figsize=(20, 10))
tree_classifier = tree_pipeline.named_steps['classifier']
plot_tree(tree_classifier, 
          feature_names=preprocessor.get_feature_names_out(),
          class_names=['Bon risque', 'Mauvais risque'],
          filled=True, 
          rounded=True, 
          fontsize=8)
plt.title('Arbre de décision (profondeur=3)')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Comparaison des performances entre GLM et arbre de décision
print("\nComparaison des performances:")
comparison_df = pd.DataFrame({
    'Modèle': ['GLM (statsmodels)', 'Decision Tree (max_depth=3)'],
    'F1-score': [f1, f1_tree],  # f1 est le F1-score du modèle GLM que vous avez calculé précédemment
    'AUC': [roc_auc, roc_auc_tree]  # roc_auc est l'AUC du GLM
})
print(comparison_df)


import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# Création du pipeline avec XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Entraînement du modèle
xgb_pipeline.fit(X_train, y_train)

# Évaluation du modèle
y_pred = xgb_pipeline.predict(X_test)
y_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

print("Rapport de classification:")
print(classification_report(y_test, y_pred))

# Métriques spécifiques
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"F1-score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")
 
 # Fonction pour récupérer les noms des caractéristiques après encodage
def get_feature_names(preprocessor, X):
    one_hot_encoder = preprocessor.transformers_[1][1]
    cat_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_features, cat_feature_names])
    return all_feature_names

# Obtention des noms des caractéristiques
feature_names = get_feature_names(preprocessor, X)
import matplotlib.pyplot as plt
import seaborn as sns

# Récupération du modèle XGBoost à partir du pipeline
xgb_model = xgb_pipeline.named_steps['classifier']

# Récupération des importances de caractéristiques
importance_scores = xgb_model.feature_importances_

# Création d'un DataFrame des importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Visualisation des 15 caractéristiques les plus importantes
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 caractéristiques les plus importantes (XGBoost native)')
plt.tight_layout()
plt.show()


from sklearn.inspection import permutation_importance

# Transformation des données de test
X_test_transformed = preprocessor.transform(X_test)

# Calcul de l'importance par permutation
result = permutation_importance(
    xgb_model, X_test_transformed, y_test,
    n_repeats=10,
    random_state=42,
    scoring='roc_auc'
)

# Création d'un DataFrame avec les résultats
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std': result.importances_std
})
perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)

print("Top 15 caractéristiques (importance par permutation):")
print(perm_importance_df.head(15))

# Création du boxplot pour chaque variable
plt.figure(figsize=(14, 10))
perm_sorted_idx = perm_importance_df.index.tolist()

# On prend les 15 premières caractéristiques pour le boxplot
perm_sorted_idx = perm_sorted_idx[:15]

# Conversion en array pour le plotting
perm_importances = result.importances[perm_sorted_idx].T
feature_names_sorted = perm_importance_df['Feature'].iloc[perm_sorted_idx].values

# Création du boxplot
plt.boxplot(perm_importances, 
           vert=False,
           labels=feature_names_sorted)
plt.title("Importance des caractéristiques par permutation")
plt.tight_layout()
plt.show()

# Alternative: barplot avec barres d'erreur
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=perm_importance_df.head(15)
)
plt.errorbar(
    x=perm_importance_df['Importance'].head(15),
    y=range(len(perm_importance_df.head(15))),
    xerr=perm_importance_df['Std'].head(15),
    fmt='none',  # Pas de marqueurs
    ecolor='black',  # Couleur des barres d'erreur
    capsize=5  # Taille des extrémités des barres d'erreur
)