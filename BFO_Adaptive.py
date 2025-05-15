#!/usr/bin/env python
# coding: utf-8

# # CI project :Natural-inspired pattern Recognition for Classification Problem 
# 
# ## Import Libraries

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split ,cross_val_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix ,classification_report,accuracy_score
from sklearn import datasets 
from sklearn.svm import SVC ,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import time
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve


# In[3]:


##Load MNIST data


# In[4]:


dataset = datasets.load_digits()
X = dataset.data
y = dataset.target


# In[5]:


dataset


# In[6]:


# Baseline Machine Learning Models: Classifiers with Default Hyperparameters


# In[7]:


## Using 3-Fold Cross-Validation
# In[21]:


#SVM
clf = SVC()
clf.fit(X,y)
scores = cross_val_score(clf, X ,y, cv=3 , scoring ='accuracy')
svc_accuracy=scores.mean()
print("Accuracy:"+ str(scores.mean()))

# In[11]:


## Traditional Parameter tunning Methods  Algorithm 1: Grid Search



# In[13]:


# SVM optimized by GridSearchCv 
from sklearn.model_selection import GridSearchCV
#Define hyperparameter Configuration space 
svm_params = {
    'C':[1,10,100],
    'kernel' :['linear','poly','rbf','sigmoid']
}
clf = SVC(gamma='scale')
grid =GridSearchCV(clf,svm_params ,cv=3 ,scoring = 'accuracy')
grid.fit(X,y)
print(grid.best_params_)
svc_accuracy_bygridsearch = grid.best_score_
print("Accuracy:"+str(grid.best_score_))



# In[15]:


##  Algorithm 2: Random Search


# In[17]:


#SVM
from scipy import stats
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
rf_params = {
    'C': stats.uniform(0,50),
    "kernel":['linear','poly','rbf','sigmoid']
}
n_iter_search=20
clf = SVC(gamma='scale')
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy')
Random.fit(X, y)
print(Random.best_params_)
svc_accuracy_byrandomsearch=Random.best_score_
print("Accuracy:"+ str(Random.best_score_))



# # Adaptive BFO Implementation (C + gamma)
# 
# This cell implements an Adaptive version of the Bacterial Foraging Optimization (BFO) algorithm, where parameters like step size or chemotaxis behavior dynamically change based on performance or generation.  
# It optimizes `C` and `gamma` hyperparameters for the SVC model using the RBF kernel.
# 
# Output: Best parameters, best accuracy, and training time.
# 

# In[23]:


# Load and scale data
dataset = load_digits()
X = dataset.data
y = dataset.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Fitness cache
fitness_cache = {}

def fitness_function(params):
    key = tuple(np.round(params, 5))  # rounded key for caching
    if key in fitness_cache:
        return fitness_cache[key]
    C, gamma = params
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
    fitness_cache[key] = score
    return score

# BFO Parameters
n_bacteria = 10
n_elimination_dispersal = 1
n_reproduction = 2
n_chemotaxis = 5
n_swim = 3
step_size_initial = 0.1  # Initial step size
step_size_decay = 0.99  # Decay factor for step size

# Parameter bounds
bounds = np.array([
    [0.01, 100],      # C
    [0.0001, 1.0]     # gamma
])

# Initialize population
np.random.seed(42)
population = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_bacteria, 2))

best_score = -np.inf
best_params = None
start_time = time.time()

for elim_disp in range(n_elimination_dispersal):
    for repro in range(n_reproduction):
        health = []
        step_size = step_size_initial  # Reset step size at the beginning of each reproduction phase

        for i in range(n_bacteria):
            position = population[i].copy()
            score = fitness_function(position)
            health_i = 0

            for j in range(n_chemotaxis):
                delta = np.random.uniform(-1, 1, 2)
                delta = delta / np.linalg.norm(delta)
                new_position = position + step_size * delta
                new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])
                new_score = fitness_function(new_position)

                swim_count = 0
                while new_score > score and swim_count < n_swim:
                    position = new_position
                    score = new_score
                    new_position = position + step_size * delta
                    new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])
                    new_score = fitness_function(new_position)
                    swim_count += 1

                health_i += score

                if score > best_score:
                    best_score = score
                    best_params = position.copy()

            health.append((health_i, i))

        # Reproduction
        health.sort(reverse=True)
        sorted_indices = [idx for _, idx in health]
        half = n_bacteria // 2
        for i in range(half):
            population[sorted_indices[half + i]] = population[sorted_indices[i]].copy()

        # Adapt step size based on fitness change
        # Reduce step size if no improvement is found during the chemotaxis phase
        if health_i == 0:
            step_size *= step_size_decay  # Adaptive step size decay

    # Elimination and dispersal
    for i in range(n_bacteria):
        if np.random.rand() < 0.25:
            population[i] = np.random.uniform(bounds[:, 0], bounds[:, 1])

end_time = time.time()

best_accuracy = best_score
# Output
print("\nBest Parameters found by AS-BFO:")
print(f"C = {best_params[0]:.5f}")
print(f"gamma = {best_params[1]:.5f}")
print(f"Best Accuracy (CV) = {best_score:.4f}")
bfo_time = end_time - start_time
print(f"Training Time = {end_time - start_time:.2f} seconds")


# # Compare SVC Optimizers: Default vs Grid vs Random vs Adaptive BFO
# 
# **Description**:  
# This cell compares four optimization methods for SVC:
# - Default parameters  
# - Grid Search  
# - Randomized Search  
# - Adaptive BFO
# 
# It records training time and accuracy for each method and summarizes the results in a structured DataFrame.
# 

# In[24]:


import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import stats

# --- Default SVC ---
start = time.time()
clf = SVC()
clf.fit(X, y)
end = time.time()
default_time = end - start

# --- Grid Search SVC ---
start = time.time()
grid = GridSearchCV(SVC(gamma='scale'), {'C': [1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}, cv=3, scoring='accuracy')
grid.fit(X, y)
end = time.time()
grid_time = end - start

# --- Random Search SVC ---
start = time.time()
random = RandomizedSearchCV(SVC(gamma='scale'),
                            {'C': stats.uniform(0, 50), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                            cv=3, n_iter=20, scoring='accuracy')
random.fit(X, y)
end = time.time()
random_time = end - start

# --- Averages for Accuracy (assuming these variables are defined beforehand) ---
default_accuracy = np.mean( svc_accuracy)
grid_accuracy = np.mean(svc_accuracy_bygridsearch)
random_accuracy = np.mean( svc_accuracy_byrandomsearch)

# Assuming best_accuracy and bfo_time are already computed from the AS-BFO optimization step:
results_df = pd.DataFrame({
    'Method': ['Default', 'Grid Search', 'Random Search', 'Adaptive BFO'],  # Updated method name
    'Training Time (s)': [default_time, grid_time, random_time, bfo_time],
    'Accuracy': [default_accuracy, grid_accuracy, random_accuracy, best_accuracy]
})

print(results_df)


# # Plot Accuracy and Training Time Comparisons
# 
# **Description**:  
# This cell visualizes performance results across methods using bar charts:
# - Accuracy comparison  
# - Training time comparison
# 
# ✅ Helps evaluate the trade-off between accuracy and computational cost for each method.
# 
# 

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt

# prepare the data
df = pd.DataFrame({
    'Method': ['Default', 'Grid Search', 'Random Search', 'Adaptive BFO'],  # Updated to Adaptive BFO
    'Training Time (s)': [default_time, grid_time, random_time, bfo_time],  # Changed bfo_time to as_bfo_time
    'Accuracy': [default_accuracy, grid_accuracy, random_accuracy, best_accuracy]
})

# plot accuracy comparison
plt.figure(figsize=(8, 4))
plt.bar(df['Method'], df['Accuracy'], color='skyblue')
plt.title('SVC Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.95, 0.98)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# plot training time comparison
plt.figure(figsize=(8, 4))
plt.bar(df['Method'], df['Training Time (s)'], color='salmon')
plt.title('SVC Training Time Comparison')
plt.ylabel('Training Time (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Validation Curve for SVC (Hyperparameter: C)
# 
# **Description**:  
# This cell plots a validation curve for SVC with RBF kernel and fixed gamma.  
# It shows how accuracy changes as the regularization parameter `C` increases, helping identify the optimal balance between bias and variance.
# 

# In[26]:


param_range = np.logspace(-3, 3, 7)
train_scores, test_scores = validation_curve(
    SVC(kernel='rbf', gamma=0.01),
    X, y,
    param_name="C",
    param_range=param_range,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean  = np.mean(test_scores,  axis=1)

plt.figure(figsize=(6,4))
plt.semilogx(param_range, train_mean, label="Train")
plt.semilogx(param_range, test_mean,  label="CV")
plt.fill_between(param_range, train_mean - np.std(train_scores, axis=1),
                 train_mean + np.std(train_scores, axis=1), alpha=0.2)
plt.fill_between(param_range, test_mean  - np.std(test_scores, axis=1),
                 test_mean  + np.std(test_scores, axis=1), alpha=0.2)
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Validation Curve for SVC')
plt.legend()
plt.grid(True)
plt.show()



# # Cross-Validation Accuracy Distribution (Boxplot)
# 
# **Description**:  
# This cell compares accuracy distributions across different SVC training methods using 5-fold stratified cross-validation.  
# Boxplots visualize how consistent each method is across folds, and highlight variance or instability in performance.
# 
# ✅ Especially useful to evaluate robustness of Adaptive BFO compared to others.
# 

# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Use the best parameters found by AS-BFO for SVC
best_C = 15.3748  # Replace with the best C parameter from AS-BFO
best_gamma = 0.0137  # Replace with the best gamma parameter from AS-BFO

methods = {
    'Default': SVC(),
    'Grid Search': SVC(C=10, kernel='rbf'),
    'Random Search': SVC(C=6.7, kernel='rbf'),
    'Adaptive BFO': SVC(C=best_C, gamma=best_gamma, kernel='rbf')  # Updated BFO method to Adaptive BFO
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_dict = {name: cross_val_score(model, X, y, cv=cv, scoring='accuracy') 
               for name, model in methods.items()}

strong_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'] 

plt.figure(figsize=(8,5))
sns.boxplot(data=list(scores_dict.values()), palette=strong_colors)
plt.xticks(range(len(scores_dict)), list(scores_dict.keys()))
plt.ylabel('Accuracy per fold')
plt.title('Cross-Validation Scores Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# === Adaptive BFO ===
def run_dynamic_bfo(X, y):
    from BFO_Adaptive import bfo_dynamic, decode_chromosome
    best_solution, best_acc, curve = bfo_dynamic(X, y)
    best_params = decode_chromosome(best_solution)
    return best_acc, best_params, curve





