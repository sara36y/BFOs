#!/usr/bin/env python
# coding: utf-8

# # CI project :Natural-inspired pattern Recognition for Classification Proplem
# 
# 
# 
# ### Hyperparameter Optimization (HPO) of Machine Learning Models
# ####  Tradional Algorithms versus Natural inspired Algorithms

# In[1]:


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


# ## Load MNIST dataset
# The MNIST database  is a large database of handwritten digits that is commonly used for training various image processing systems. The MNIST database has a training set of 60,000 examples, and a test set of 10,000 examples.

# In[2]:


dataset = datasets.load_digits()
X = dataset.data
y = dataset.target


# In[3]:


dataset


# ## Baseline Machine Learning Models: Classifiers with Default Hyperparameters

# ### Using 3-Fold Cross-Validation

# In[4]:


#SVM
clf = SVC()
clf.fit(X,y)
scores = cross_val_score(clf, X ,y, cv=3 , scoring ='accuracy')
print("Accuracy:"+ str(scores.mean()))


# ## Tradition Parameter tunning Methods  Algorithm 1: Grid Search
# Search all the given hyper-parameter configurations
# 
# **Advantages:**
# * Simple implementation.  
# 
# **Disadvantages:**  
# * Time-consuming,
# * Only efficient with categorical HPs.

# In[5]:


# SVM optimized by GridSearchCv
from sklearn.model_selection import GridSearchCV
#Define hyperparameter Configuration space
svm_params = {
    'C':[1, 10, 100],
    'kernel' :['linear','poly','rbf','sigmoid']
}
clf = SVC(gamma='scale')
grid =GridSearchCV(clf,svm_params ,cv=3 ,scoring = 'accuracy')
grid.fit(X,y)
print(grid.best_params_)
print("Accuracy:"+str(grid.best_score_))
svc_accuracy_bygridsearch = grid.best_score_
svc_params_bygridsearch = grid.best_params_


# ##  Algorithm 2: Random Search
# Randomly search hyper-parameter combinations in the search space
# 
# **Advantages:**
# * More efficient than GS.
# * Enable parallelization.
# 
# **Disadvantages:**  
# * Not consider previous results.
# * Not efficient with conditional HPs.

# In[6]:


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
print("Accuracy:"+ str(Random.best_score_))
svc_accuracy_byrandomsearch = Random.best_score_
svc_params_byrandomsearch = Random.best_params_


# ## Define local search BFO  helpers
# Define helper functions for local search Bacterial Foraging Optimization , including binary chromosome decoding, population initialization, fitness evaluation, and bit-flipping (mutation).
# 

# In[7]:


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Decode binary chromosome to SVM parameters
def decode_chromosome(chromosome):
    c_bits = chromosome[:10]
    kernel_bits = chromosome[10:]

    c_value = int("".join(map(str, c_bits)), 2)
    c_real = 0.1 + (1000 - 0.1) * (c_value / (2**10 - 1))

    kernel_idx = int("".join(map(str, kernel_bits)), 2)
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_value = kernel_options[kernel_idx % 4]

    return {'C': c_real, 'kernel': kernel_value}

# Evaluate chromosome
def evaluate(chromosome, X, y):
    params = decode_chromosome(chromosome)
    clf = SVC(C=params['C'], kernel=params['kernel'], gamma='scale')
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    return scores.mean()

# Initialize population
def initialize_population(pop_size, n_bits):
    return [np.random.randint(0, 2, n_bits).tolist() for _ in range(pop_size)]

# Bit flip mutation
def flip_bit(chromosome, prob=0.1):
    return [(1 - bit if np.random.rand() < prob else bit) for bit in chromosome]

# Local search
def local_search(chromosome, X, y):
    best_chromosome = chromosome.copy()
    best_fitness = evaluate(best_chromosome, X, y)

    for i in range(len(chromosome)):
        neighbor = chromosome.copy()
        neighbor[i] = 1 - neighbor[i]
        fitness = evaluate(neighbor, X, y)

        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = neighbor

    return best_chromosome, best_fitness



# ## Run local search BFO  optimization
# Execute the local search BFO  algorithm to optimize SVM hyperparameters (C and kernel). Track best accuracy over generations.
# 

# In[8]:


# BFO with Local Search
def bfo_with_local_search(X, y, n_gen=20, pop_size=10):
    n_bits = 12
    population = initialize_population(pop_size, n_bits)
    fitness_over_gens = []

    best_solution = None
    best_fitness = 0

    for gen in range(n_gen):
        new_population = []
        fitnesses = []

        for bacterium in population:
            mutated = flip_bit(bacterium, prob=0.1)
            mutated_fitness = evaluate(mutated, X, y)
            original_fitness = evaluate(bacterium, X, y)

            selected = mutated if mutated_fitness > original_fitness else bacterium
            selected_fitness = max(mutated_fitness, original_fitness)

            # Local search step
            improved, improved_fitness = local_search(selected, X, y)

            new_population.append(improved)
            fitnesses.append(improved_fitness)

        population = new_population
        gen_best = max(fitnesses)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_solution = population[fitnesses.index(gen_best)]

        fitness_over_gens.append(gen_best)
        print(f"Generation {gen+1} - Best Accuracy: {gen_best:.4f}")

    return best_solution, best_fitness, fitness_over_gens



# ## Decode best local search BFO result 
# Extract the best SVM parameters from the local search BFO result and print the best accuracy.
# 

# In[9]:


best_solution_local, svc_accuracy_bfols, fitness_curve_ls = bfo_with_local_search(X, y)
svc_params_bfols = decode_chromosome(best_solution_local)

print("Best Params (BFO + Local Search):", svc_params_bfols)
print("Best Accuracy (BFO + Local Search):", svc_accuracy_bfols)


# ## Accuracy comparison plot
# Compare the accuracies of GridSearchCV, RandomizedSearchCV, and local BFO using a bar chart.
# 

# In[10]:


plt.plot(fitness_curve_ls)
plt.xlabel("Generation")
plt.ylabel("Best Accuracy")
plt.title("Local Search BFO - Fitness Over Generations")
plt.grid()
plt.show()


# ## Print best parameters and accuracies
#  Display a side-by-side comparison of the best parameters and accuracies obtained from each optimization method.
# 

# In[11]:


print("=== Accuracy Comparison ===")
print(f"GridSearchCV Accuracy:     {svc_accuracy_bygridsearch:.4f} | Params: {svc_params_bygridsearch}")
print(f"RandomSearchCV Accuracy:   {svc_accuracy_byrandomsearch:.4f} | Params: {svc_params_byrandomsearch}")
print(f"local search BFO Accuracy:       {svc_accuracy_bfols:.4f} | Params: {svc_params_bfols}")


import matplotlib.pyplot as plt

results = {
    'GridSearchCV': svc_accuracy_bygridsearch,
    'RandomSearchCV': svc_accuracy_byrandomsearch,
    'local search BFO': svc_accuracy_bfols
}

labels = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color='skyblue')
plt.ylabel('Accuracy')
plt.ylim(min(accuracies) - 0.001, max(accuracies) + 0.001)
plt.title('Accuracy Comparison: GridSearchCV vs RandomSearchCV vs local search BFO')
plt.grid(axis='y')
plt.show()


# In[13]:


# === Local Search BFO ===
def run_bfo_local_search(X, y):
    from BFO_Localsearch import bfo_with_local_search, decode_chromosome
    best_solution, best_acc, curve = bfo_with_local_search(X, y)
    best_params = decode_chromosome(best_solution)
    return best_acc, best_params, curve






