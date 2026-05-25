import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df =pd.read_csv(r"C:\Users\hp\TPs\ML\heart_desease_data.csv")

numerical = ['age','trestbps','chol','thalach','oldpeak','ca']
categorical = ['genre','cp','fbs','restecg','exang','slope','thal','target'] 

#checking missing data 
missing_data = df.isnull().sum()
print(missing_data)

# checking for duplicate rows
duplicates = df[df.duplicated()]

if not duplicates.empty:
    print("duplicate rows found :")
    print(duplicates)
else:
    print("no duplicate rows found!")
    
    
#checking for outliers
for column in numerical:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"outliers dans {column} : {len(outliers)}")
    
#data exploration
print(df.describe())


# create histograms for numerical columns
for i, col in enumerate(numerical, 1):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df[col])
    plt.title(f'Histogramme de {col}')
    plt.show()
    
# create box plots for numerical columns
for i, col in enumerate(numerical, 1):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[col])
    plt.title(f'Box Plot de {col}')
    plt.show()
    
# create quantile plots for numerical columns
for i, col in enumerate(numerical, 1):
    plt.figure(figsize=(6, 4))
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot de {col}')
    plt.show()


# create count plots for categorical columns

# 1. define mappings for categorical values
mappings = {
    'genre': {0: 'Male', 1: 'Female'},
    'cp': {0: 'Asymptomatic', 1: 'Non-Anginal Pain', 2: 'Atypical Angina', 3: 'Typical Angina'},
    'fbs': {0: 'No', 1: 'Yes'},
    'restecg': {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'},
    'exang': {0: 'No', 1: 'Yes'},
    'slope': {0: 'Upsloping', 1: 'Downsloping', 2: 'Flat'},
    'ca': {0: '0 Vessels', 1: '1 Vessel', 2: '2 Vessels', 3: '3 Vessels'},
    'thal': {0: 'Fixed Defect', 1: 'Reversible Defect', 2: 'Normal', 3: 'Unknown'},
    'target': {0: 'No Heart Disease', 1: 'Heart Disease'}
}

# 2. apply mappings to categorical columns
for col in categorical:
    df[col] = df[col].map(mappings[col])
    
# 3. create the bar plots
for i, col in enumerate(categorical, 1):
    sns.countplot(x=df[col], palette="Set2")
    plt.xticks(rotation=30) 
    plt.title(f"Distribution of {col}")
    plt.show()
    

# create pie charts for categorical columns
for i, col in enumerate(categorical, 1):
    plt.figure(figsize=(6, 6))
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap="Pastel1") 
    plt.title(f"Répartition de {col}")
    plt.ylabel("")
    plt.show()

# histogram of a continuous variable (age) concerning different values of a categorical variable (target) 
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', kde=True, element='step', hue='cp', palette='viridis')
plt.title("Distribution de l'âge selon la présence de maladie")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.legend(title='Maladie', labels=['Absente', 'Présente'])

# correlation matrix with heatmap
correlation_matrix = df[numerical].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation des variables numériques")


# correlation matrix with pairplot 
sns.pairplot(df, vars=numerical, hue="target", palette="coolwarm", diag_kind="kde")
plt.show()

# barplot of a categorical variable (genre) according to different values of a categorical variable (target)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='genre', hue='target', palette='Set2')
plt.title("Répartition du genre selon la présence de maladie")
plt.xlabel("Genre (0=Femme, 1=Homme)")
plt.ylabel("Nombre de patients")
plt.legend(title='Maladie', labels=['Absente', 'Présente'])

# mosaïc plot of categorical variables (type of chest pain (cp) and presence of disease (target)) 
plt.figure(figsize=(12, 8))
mosaic(df, ['cp', 'target'], title='Relation entre type de douleur et maladie', gap=0.02)
plt.xlabel("Type de douleur (cp)")
plt.ylabel("Maladie (target)")
   
# boxplot of a continuous variable (chol) concerning different values of a categorical variable (target) 
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='cp', y='chol', palette='pastel')
plt.title("Distribution du cholestérol selon le type de douleur thoracique")
plt.xlabel("Type de douleur (cp)")
plt.ylabel("Cholestérol (mg/dL)")
plt.xticks([0, 1, 2, 3], ['Typique', 'Atypique', 'Non-angineuse', 'Asymptomatique'])
plt.show()


#tp2 part

# 1. split data into train and test with train_test_split from sklearn

# Splitting the dataset
X = df.drop(columns=['target'])  # features
y = df['target']  # target variable

X = pd.get_dummies(X, drop_first=True)  # one-hot encode all string columns

# Splitting into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. split data into train and test with a custom function 
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    # shuffle indices
    indices = np.random.permutation(len(X))  
    test_size = int(len(X) * test_size)
    
    #determin the test size and the train size of the dataset
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)


# 3. run knn algorithm for k=1
# create a KNN modek instance with n_neighbors=1
knn = KNeighborsClassifier(n_neighbors=1)
#fit the model to the training data
knn.fit(X_train, y_train)

# use predict method to predict values using knn model and x_test
y_pred = knn.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy with k=1: {accuracy:.2f}')

# 4. best k value 
error_rates = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, y_pred))

# Plotting error vs k
plt.figure(figsize=(10,6))
plt.plot(k_values, error_rates, marker='o', linestyle='dashed', color='b')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K Value')
plt.show()
