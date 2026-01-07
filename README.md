# Version-Finale

# DIABETES ANALYSIS

# A.LARHLIMI

## HLAL KAWTAR

<img src="image7.png" style="height:540px;margin-right:393px"/>

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année


### 1. Le Problème (Business Case)
Dans le domaine médical, le suivi de l’évolution d’une maladie chronique est souvent complexe, car il dépend de nombreux facteurs cliniques et biologiques difficilement interprétables conjointement par un médecin seul. La variabilité inter‑patient et la charge de travail élevée peuvent conduire à une sous‑estimation ou une surestimation de la gravité réelle de la maladie.

- Objectif : 
Concevoir un modèle de prédiction de la progression de la maladie (variable cible continue du dataset) à partir des caractéristiques cliniques et biologiques du patient, afin d’aider le médecin à anticiper l’évolution et adapter le traitement.

- Enjeu critique : 
Même si la cible est continue, une erreur de prédiction n’a pas le même impact clinique selon qu’elle sous‑estime ou surestime la gravité de la maladie.​

 Une surestimation de la progression (prédire une maladie plus grave qu’elle ne l’est vraiment) peut entraîner des traitements plus lourds, des effets secondaires inutiles et des coûts supplémentaires pour le système de santé.
 Une sous‑estimation de la progression (prédire une maladie moins avancée qu’en réalité) peut retarder la mise en place d’un traitement adapté, aggravant le pronostic du patient et augmentant le risque de complications sévères.

### Les Données (L'Input)
Dans ce projet, les données proviennent d’un dataset médical réel décrivant la progression d’une maladie chronique à partir de mesures cliniques et biologiques de patients. Le jeu de données contient 442 observations et 10 variables explicatives normalisées, plus une variable cible continue représentant une mesure de progression de la maladie.

- X (Features) : 10 colonnes correspondant à des caractéristiques numériques standardisées du patient, telles que l’âge, le sexe codé, l’indice de masse corporelle (bmi), la pression sanguine (bp) et plusieurs mesures biologiques (s1 à s6). Ces variables sont déjà centrées‑réduites, ce qui facilite l’entraînement de modèles de régression.

- y (Target) : une variable continue représentant un score de progression de la maladie, utilisé comme indicateur de gravité ou d’avancement. Plus la valeur est élevée, plus la progression de la maladie est importante.

### 2. Le Code Python 

```python
# ==============================================================================
#  DIABETES ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTATION DES BIBLIOTHÈQUES
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modules Scikit-Learn spécifiques
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor 

# Changed from RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score 

# Changed from accuracy_score, classification_report, confusion_matrix

# Configuration pour des graphiques plus esthétiques
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore') 

# Pour garder la sortie propre

print("1. Bibliothèques importées avec succès.\n")

# ------------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES
# ------------------------------------------------------------------------------

# Chargement du dataset depuis Scikit-Learn
data = load_diabetes()

# Création du DataFrame Pandas
# data.data contient les features, data.target contient une mesure quantitative de la progression de la maladie
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"2. Données chargées. Taille du dataset : {df.shape}")
print(f"   Le target est une variable continue (mesure de progression de la maladie).\n")

# ------------------------------------------------------------------------------
# 3. SIMULATION DE "DONNÉES SALES" (Pour l'exercice)
# ------------------------------------------------------------------------------

# Dans la vraie vie, les données sont rarement parfaites.
# Nous allons introduire artificiellement des valeurs manquantes (NaN) dans 5% des données.
print("3. Introduction artificielle de valeurs manquantes (NaN)...")

np.random.seed(42) # Pour la reproductibilité
mask = np.random.random(df.shape) < 0.05 # Masque de 5%

# On applique les NaN partout sauf sur la colonne 'target' (qu'on ne veut pas abîmer ici)
features_columns = df.columns[:-1]
df_dirty = df.copy()
for col in features_columns:
    df_dirty.loc[df_dirty.sample(frac=0.05).index, col] = np.nan

print(f"   Nombre total de valeurs manquantes générées : {df_dirty.isnull().sum().sum()}\n")

# ------------------------------------------------------------------------------
# 4. NETTOYAGE ET PRÉPARATION (Data Wrangling)
# ------------------------------------------------------------------------------

print("4. Nettoyage des données...")

# Séparation Features (X) et Target (y) AVANT le nettoyage pour éviter les fuites de données
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# Imputation : Remplacer les NaN par la MOYENNE de la colonne
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# On remet sous forme de DataFrame pour garder les noms de colonnes (plus propre)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print("   Imputation terminée (les NaN ont été remplacés par la moyenne).")
print(f"   Valeurs manquantes restantes : {X_clean.isnull().sum().sum()}\n")

# ------------------------------------------------------------------------------
# 5. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)

# ------------------------------------------------------------------------------
print("5. Analyse Exploratoire (EDA)...")

# A. Aperçu statistique
print("   Statistiques descriptives (premières 5 colonnes) :")
print(X_clean.iloc[:, :5].describe())

# B. Visualisation 1 : Distribution d'une feature clé
plt.figure(figsize=(10, 5))
feature_to_plot = 'bmi' # Changed from 'mean radius' to 'bmi'
sns.histplot(data=df, x=feature_to_plot, hue='target', kde=True, element="step")
plt.title(f"Distribution de '{feature_to_plot}' selon le diagnostic") # Removed classification labels
plt.show()

# C. Visualisation 2 : Heatmap de corrélation (sur les 10 premières variables pour la lisibilité)
plt.figure(figsize=(10, 8))
correlation_matrix = X_clean.iloc[:, :10].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation (Top 10 Features)")
plt.show()

# ------------------------------------------------------------------------------
# 6. SÉPARATION DES DONNÉES (Train / Test Split)
# ------------------------------------------------------------------------------

# On garde 20% des données pour le test final
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

print(f"\n6. Séparation effectuée :")
print(f"   Entraînement : {X_train.shape[0]} échantillons")
print(f"   Test : {X_test.shape[0]} échantillons\n")

# ------------------------------------------------------------------------------
# 7. MODÉLISATION (Machine Learning)
# ------------------------------------------------------------------------------

print("7. Entraînement du modèle (Random Forest Regressor)...") # Updated model name

# Initialisation du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42) # Changed to Regressor

# Entraînement sur les données d'entraînement uniquement
model.fit(X_train, y_train)
print("   Modèle entraîné avec succès.\n")

# ------------------------------------------------------------------------------
# 8. ÉVALUATION ET PERFORMANCE
# ------------------------------------------------------------------------------

print("8. Évaluation des performances...")

# Prédictions sur le jeu de test (données jamais vues par le modèle)
y_pred = model.predict(X_test)

# A. Mean Squared Error (Erreur quadratique moyenne)
mse = mean_squared_error(y_test, y_pred)
print(f"   >>> Mean Squared Error : {mse:.2f}")

# B. R2 Score (Coefficient de détermination)
r2 = r2_score(y_test, y_pred)
print(f"   >>> R2 Score : {r2:.2f}")

# C. Visualisation des prédictions vs. réalité (pour la régression)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions du modèle vs. Valeurs réelles')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') # Ligne idéale
plt.show()

print("\n--- FIN DU SCRIPT ---")
```
## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

Dans notre projet, des valeurs manquantes (NaN) ont été introduites artificiellement dans les variables explicatives pour simuler un cas réel de données incomplètes, puis totalement imputées.  Au total, 220 valeurs manquantes ont été générées puis remplacées, et le dataset final ne contient plus aucun NaN avant la phase de modélisation.

### Le problème mathématique du « vide »

La plupart des algorithmes de modélisation utilisés en machine learning reposent sur des opérations d’algèbre linéaire (produits matriciels, inverses, calculs de distances, etc.). Une seule valeur NaN dans une matrice peut rendre ces opérations impossibles à évaluer numériquement, ce qui fait échouer le calcul ou donne des résultats non définis. Les bibliothèques standards de calcul scientifique (NumPy, scikit‑learn, etc.) exigent donc en pratique qu’aucune valeur NaN ne subsiste dans les données d’entrée des modèles.  

### La mécanique de l’imputation

Pour traiter ce problème, une étape d’imputation systématique des valeurs manquantes a été mise en place sur toutes les colonnes numériques à l’aide d’un `SimpleImputer(strategy='mean')`.
- **Apprentissage (fit)** : pour chaque feature, l’imputeur parcourt la colonne et calcule la moyenne μ uniquement sur les valeurs observées.  
- **Transformation (transform)** : lors du passage suivant, chaque NaN détecté dans cette colonne est remplacé par la moyenne μ préalablement stockée, ce qui permet de reconstruire une matrice complète, compatible avec les algorithmes de régression.

Cette approche par la moyenne est simple, stable numériquement et conserve la taille du dataset, au prix d’un lissage de la variabilité réelle des données.  

### Le coin de l’expert : Data Leakage

Dans ce notebook pédagogique, l’imputation a été effectuée sur l’ensemble du dataset avant la séparation en ensembles d’entraînement et de test.  Cette pratique est tolérable en contexte académique, mais introduit en production un risque de **Data Leakage** (fuite d’information), car les statistiques de nettoyage (ici, les moyennes) utilisent indirectement des informations du futur jeu de test.

La **bonne pratique industrielle** consiste à :  
1. Séparer d’abord les données en `Train` et `Test`.  
2. Ajuster l’imputeur (fit) uniquement sur le `Train` pour calculer les moyennes des colonnes.  
3. Appliquer ensuite la transformation (transform) avec ces mêmes moyennes sur le `Train` et sur le `Test`.  

Ainsi, le modèle ne voit jamais, même indirectement, l’information contenue dans le set de test au moment du nettoyage et de l’apprentissage, ce qui garantit une évaluation plus honnête de ses performances.

## 4. Analyse Approfondie : Exploration (EDA)

L'exploration des données (EDA) constitue l'étape de "profilage" du dataset, permettant de comprendre la structure statistique des variables avant modélisation.  Les statistiques descriptives ont été calculées sur les 10 features normalisées, révélant des distributions centrées autour de 0 avec des écarts-types homogènes autour de 0.046.

### Décrypter .describe()

Les statistiques de base fournissent des insights cruciaux sur chaque feature :[1]
- **Mean (Moyenne) vs 50% (Médiane)** : Les moyennes sont proches de 0 (ex. age : 0.000505, bmi : 0.000197), tout comme les médianes (ex. age : 0.001751, bmi : -0.005128). Cette symétrie suggère des distributions **non asymétriques** (non skewed), sans valeurs extrêmes tirant fortement la moyenne.  
- **Std (Écart-type)** : Tous les std sont similaires (~0.046), indiquant une **largeur de distribution cohérente** entre features. Aucun std proche de 0, donc toutes les variables portent de l'information (pas de constantes inutiles).
- **Extrêmes** : Les min/max montrent une variabilité raisonnable (ex. s1 : min -0.127 à max 0.154), confirmant l'efficacité de la normalisation centrée-réduite.

### La multicollinéarité (Le problème de la redondance)

Bien que non explicitement visualisée dans le notebook, une analyse de corrélation serait pertinente pour détecter la multicollinéarité entre les 10 features biologiques et cliniques.  Géométriquement, des mesures liées comme bmi et bp pourraient présenter des corrélations élevées (>0.8), rendant les features redondantes.

**Impact ML** :  
- Pour des modèles ensemblistes comme Random Forest, la multicollinéarité est tolérée (arbre de décision gère la redondance).  
- Pour la régression linéaire (adaptée à notre cible continue), elle déstabilise les coefficients : le modèle peine à attribuer le "poids" prédictif à une feature unique parmi des variables corrélées, augmentant la variance des prédictions.

En pratique, une matrice de corrélation (heatmap) ou VIF (Variance Inflation Factor) permettrait d'identifier et éliminer les features les plus redondantes avant modélisation.

## 5. Méthodologie du split (Train/Test)

Dans ce projet, la séparation des données en ensembles d’entraînement et de test sert à évaluer la capacité du modèle à **généraliser** sur de nouveaux patients jamais vus pendant l’apprentissage. Le but n’est pas de mémoriser les exemples passés, mais de construire une relation robuste entre les variables cliniques et la progression de la maladie, capable de se transférer au futur.

### Le concept : garantie de généralisation

Si l’on entraînait et évaluait le modèle sur les mêmes données, on mesurerait seulement sa capacité à « réciter » les cas du passé, pas à prédire correctement de nouveaux cas.  
En réservant un sous‑ensemble indépendant pour le test, on obtient une estimation plus honnête des performances réelles en situation clinique, ce qui est essentiel avant d’envisager un déploiement auprès de médecins.

### Les paramètres du split

Une séparation typique pour ce type de dataset est :  
```python
train_test_split(test_size=0.2, random_state=42)
```
- **Ratio 80/20 (Principe de Pareto)** : environ 80 % des patients sont utilisés pour apprendre les motifs complexes entre les features et la progression de la maladie (Train), et 20 % sont conservés pour mesurer la performance sur des données « nouvelles » (Test). Ce compromis laisse suffisamment d’exemples pour l’apprentissage tout en gardant un test assez grand pour que la métrique soit statistiquement exploitable.  
- **Reproductibilité (`random_state=42`)** : le tirage des patients dans Train et Test repose sur un générateur pseudo‑aléatoire. Fixer la graine (42) garantit que chaque exécution produira exactement la même répartition des patients. Cela permet à un autre chercheur, sur une autre machine, de reproduire à l’identique les résultats du modèle, condition indispensable à une validation scientifique rigoureuse.

## 6. Focus théorique : Random Forest 🌲

Le Random Forest est souvent considéré comme un « couteau suisse » du Machine Learning car il est robuste, performant dès le premier essai, gère bien les non‑linéarités et les interactions entre variables, et nécessite peu de préparation des données (peu sensible au scaling, aux distributions bizarres, et assez tolérant à la multicollinéarité). Il s’adapte aussi bien à la classification qu’à la régression, ce qui en fait un choix par défaut très utilisé en pratique.

### A. La faiblesse de l’individu (Arbre de décision)

Un arbre de décision unique apprend en posant des questions successives du type « si telle feature > seuil alors aller à gauche, sinon à droite », jusqu’à aboutir à des feuilles qui donnent une prédiction.  
Problème : il a une **variance très élevée**. Il peut facilement sur‑apprendre le bruit, par exemple créer une règle très spécifique pour un patient très atypique, au lieu de capturer le vrai motif général. Un changement léger dans les données d’entraînement peut complètement changer la forme de l’arbre.

### B. La force du groupe (Bagging)

Le Random Forest construit non pas un, mais des dizaines voire des centaines d’arbres, chacun entraîné dans des conditions légèrement différentes.  
Deux sources de « chaos contrôlé » sont utilisées :  
- **Bootstrapping (échantillons différents)** : chaque arbre est entraîné sur un échantillon tiré avec remise du dataset (certains patients sont répétés, d’autres absents), ce qui donne à chaque arbre une « expérience » différente.  
- **Feature randomness (colonnes différentes)** : à chaque split, l’arbre ne voit qu’un sous‑ensemble aléatoire des features, ce qui l’oblige à utiliser aussi des variables moins évidentes au lieu de se reposer toujours sur les plus fortes.  

Cette double randomisation réduit fortement la corrélation entre arbres et donc la variance globale du modèle.

### C. Le consensus (Vote ou moyenne)

Lorsqu’un nouveau patient arrive :  
- En **classification**, chaque arbre donne une classe (par exemple malade / sain) et la forêt prend la décision finale par **vote majoritaire**.  
- En **régression**, chaque arbre donne une valeur numérique et la forêt renvoie la **moyenne** des prédictions.  

Parfait, tu suis exactement la structure du corrigé, mais dans ton cas on est en **régression** (cible continue), pas en classification.

On va donc adapter la partie **“Évaluation”** à un modèle de **régression** (par ex. RandomForestRegressor).

***

## 7. Analyse Approfondie : Évaluation

Comment lire les résultats comme un pro, quand la cible est continue (progression de maladie) ?

### A. Pas de matrice de confusion en régression

La matrice de confusion suppose des **classes** (malade / sain).  
Ici, on prédit un **score numérique** de progression, donc on ne compte pas TP, FP, FN, TN.  
On mesure plutôt **l’écart** entre la vraie progression et la progression prédite.

### B. Les métriques avancées en régression

Les principales métriques pour juger la qualité du modèle sont :

- **MSE (Mean Squared Error)** : moyenne des carrés des erreurs \((y_{\text{réel}} - y_{\text{prédit}})^2\).  
  Plus le MSE est bas, plus le modèle colle globalement aux valeurs réelles.  
- **RMSE (Root Mean Squared Error)** : racine carrée du MSE, dans la même unité que la cible.  
  Interprétation plus intuitive : “erreur moyenne typique” sur le score de progression.  
- **MAE (Mean Absolute Error)** : moyenne des erreurs absolues \(|y_{\text{réel}} - y_{\text{prédit}}|\).  
  Plus robuste aux valeurs extrêmes : donne l’erreur moyenne en “points de progression”.
- **\(R^2\) (Coefficient de détermination)** : varie en général entre 0 et 1.  
  - 0 : le modèle ne fait pas mieux qu’une prédiction constante (la moyenne).  
  - 1 : prédiction parfaite.  
  C’est une mesure de la **proportion de variance expliquée** par le modèle.

Dans un contexte médical, la question clé est :  
> « Dans quelle mesure le modèle se trompe sur la progression, et ces erreurs sont‑elles cliniquement acceptables ? »

Par exemple :  
- Un **RMSE faible** signifie que, en moyenne, le modèle se trompe peu sur le score de progression.  
- Un **\(R^2\) élevé** signifie que le modèle capte bien la relation entre les variables cliniques/biologiques et l’évolution de la maladie.

## Conclusion du projet

Ce rapport montre que la Data Science ne s’arrête pas à `model.fit()`.  
C’est une chaîne de décisions logiques où la **compréhension du métier médical** dicte :  
- le choix du **type de modèle** (ici un Random Forest de régression pour la robustesse face au bruit et aux non‑linéarités),  
- et le choix des **métriques d’évaluation** (MSE, RMSE, MAE, \(R^2\)) pour quantifier de façon honnête la qualité des prédictions de progression.

L’enjeu final n’est pas seulement d’optimiser une métrique mathématique, mais de savoir si l’erreur résiduelle du modèle est compatible avec une **prise de décision clinique sûre** (anticiper une aggravation, ajuster un traitement, surveiller un patient plus étroitement).
