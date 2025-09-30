import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

#data cleaning
df = pd.read_csv('asd.csv')
print(df.shape)
print(df.columns)

print(df.info())
print(df.describe())

missing_vals = print(df.isnull().sum())

df = df.drop('Unnamed: 10', axis=1)
df = df.drop('Child_ID', axis=1)
df['Target'] = df['Diagnosed_ASD']
df = df.drop('Diagnosed_ASD', axis=1)

columns = ['Social_Interaction_Score', 'Communication_Score', 'Repetitive_Behavior_Score', 'Age' ]
df_filtered = df[(df[columns] >= 0).all(axis=1)]

print(df.columns)

print(df.to_string())

#data visualization 
MaleVsFemale = df['Gender'].value_counts()  
plt.figure(figsize=(6,6))
plt.pie(MaleVsFemale, labels= ['Female','Male'], startangle=90)
plt.title('Gender Distribution in Dataset')
plt.show()

Desired_Columns_For_Charts = ['Language_Delay', 'Family_ASD_History', 'Jaundice', 'Target']
Desired_Columns_Translation = df[Desired_Columns_For_Charts].replace({'Yes': 1, 'No': 0})  

plt.figure(figsize=(12,10))
sns.heatmap(Desired_Columns_Translation.corr(), annot=True, cmap='coolwarm')
plt.title('The Effect of Language, Family History, Jaundice & ASD')
plt.show()

Desired_Score_Columns_For_Charts = ['Social_Interaction_Score', 'Communication_Score', 'Repetitive_Behavior_Score', 'Target']
Desired_Num_Score_Columns_Translation = df[Desired_Columns_For_Charts].replace({'Yes': 1, 'No': 0})  

plt.figure(figsize=(12,10))
sns.heatmap(Desired_Num_Score_Columns_Translation.corr(), annot=True, cmap='coolwarm')
plt.title('Social Scores & ASD')
plt.show()

plt.figure(figsize=(12, 10))
sns.countplot(x='Target', data=df)
plt.ylabel("Candidates")
plt.title('Distribution of ASD vs TD')
plt.show()

plt.figure(figsize=(12, 10))
sns.countplot(x='Gender', hue='Target', data=df)
plt.ylabel("Candidates")
plt.title('ASD Cases by Gender')
plt.show()

plt.figure(figsize=(12, 10))
sns.histplot(data=df, x='Age', hue='Target', kde=True, bins=10)
plt.title('Age Distribution by ASD Diagnosis')
plt.xlabel("Age (in months)")
plt.ylabel("Candidates")
plt.show()


for col in ['Language_Delay', 'Family_ASD_History', 'Jaundice']:
    plt.figure(figsize=(12, 10))
    sns.countplot(x=col, hue='Target', data=df)
    plt.ylabel("Candidates")
    plt.title(f'ASD Cases by {col}')
    plt.show()

for score in ['Social_Interaction_Score','Communication_Score','Repetitive_Behavior_Score']:
    plt.figure(figsize=(12, 10))
    sns.boxplot(x='Target', y=score, data=df)
    plt.title(f'{score} by ASD Diagnosis')
    plt.show()

#for training and testing 
#normalization
scaler= MinMaxScaler()
numeric_cells= ['Social_Interaction_Score', 'Communication_Score', 'Repetitive_Behavior_Score', 'Age']
df[[col +'_norm' for col in numeric_cells]] = scaler.fit_transform(df[numeric_cells])
df[[col +'_norm' for col in ['Social_Interaction_Score', 'Communication_Score', 'Repetitive_Behavior_Score', 'Age'] ]].head()

#normalized heatmap
combined_df = pd.concat([df[['Social_Interaction_Score_norm']], df['Communication_Score_norm'],df['Repetitive_Behavior_Score_norm'] ,df[['Age']], Desired_Columns_Translation], axis=1)
plt.figure(figsize=(12,10))
sns.heatmap(combined_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#logistic regression 
x=df[['Social_Interaction_Score_norm', 'Communication_Score_norm', 'Repetitive_Behavior_Score_norm', 'Age_norm', 'Language_Delay', 'Family_ASD_History', 'Jaundice']]
y=df['Target']
categorical_cols = ['Language_Delay', 'Family_ASD_History', 'Jaundice']
x=pd.get_dummies(x, columns=categorical_cols, drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=42)
model=LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(x_train,y_train)

train_accuracy=model.score(x_train,y_train)
test_accuracy=model.score(x_test,y_test)
print(f"Training Accuracy:{train_accuracy:.2f}")
print(f"Testing Accuracy:{test_accuracy:.2f}")
y_pred=model.predict(x_test)

#classification report 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

report=classification_report(y_test,y_pred)
print("\nClassification Report:\n",report)

#random forest 
model = RandomForestClassifier(class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#classification report 
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

classifiers = {
    'Decision Tree': DecisionTreeClassifier( max_depth=100,random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced',random_state=42),
    'SVM': SVC(probability=True,random_state=42),
    'Naive Bayes': GaussianNB(), 
    'KNN': KNeighborsClassifier(),
    'Bagging': BaggingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42)
}

for name, model in classifiers.items():
    print(f"\n=== {name} ===")

    if name in ['SVM', 'MLP', 'KNN']: 
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(x_train, y_train_encoded)
        y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test_encoded, y_pred)
    acc = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_)

    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:\n", report)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.show()


y_pred_proba = model.predict_proba(x_test)[:, 1]  
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Yes')
roc_auc = auc(fpr, tpr)
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for ASD Classification')
plt.legend()
plt.show()