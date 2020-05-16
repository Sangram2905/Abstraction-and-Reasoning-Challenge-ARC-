# %% [markdown]
# # **Introduction**
# 
# **Abstraction and Reasoning Challenge** :  As a beginner to look up this Challenge and comparing to our High school ABSTRACT REASONING TESTS Questions.
# 
# In this note book I like to solve this problem and it will show how we can use data visualization & Machine learning in python
# 
# 
# 

# %% [markdown]
# # **Contents:**
# 
# **Preparing the ground**
# * About the Data #Code to exract some formats
# 
# * Import libraries and define hyperparameters
# * Load the ARC data
# 
# **Basic exploration**
# * Look at few train/test input/output pairs
# * Number frequency
# * Matrix mean values
# * Matrix heights
# * Matrix widths
# * Height vs. Width
# 
# **Data processing**
# 
# 1. Modeling (Under process)
# 1. Training and postprocessing (Under process)
# 1. Loss (MAE) (Under process)
# 1. Backpropagation and optimization (Under process)
# 
# **Submission**
# 
# **Ending note**
# 

# %% [markdown]
# # Preparing the ground

# %% [markdown]
# **About the Data**

# %% [markdown]
# #The task is consist of JSON file where each JSON file comes with following pairs:
# 
# Task dictionary -> Test & Train list -> Train & Train Dictionary ->Input & output list-> Color data list 
# 1.	We can divide the data in following to simplify and Solvable formats
# 2.	Maximum grid is divided in  1X1 to 30X30 Rows X Columns 
# 3.	Grid also can divide in 3 parts
# 
#     a.	Small input big output
#     
#     b.	Big Input Small output
#     
#     c.	Equal input equal output
#     
#     d.	Simple Input to (Simple, Medium or Hard) Output
#     
#     e.	Medium Input to (Simple, Medium or Hard) Output
#    
#     f.	Hard Input to (Simple, Medium or Hard) Output
#    
# 4.	Above Complex patterns calculated by grid size.
# 5.	Columns & rows can be sub divided as:
#     a.	Even rows even columns
#     b.	Even rows odd columns
#     c.	Odd rows odd columns
#     d.	Odd rows even columns
# 
# 6.	Number of pairs in one Train or test sets.
# 
# #7.	The core data “Color” is of type 10
# 
# 8.	The direction of color movement 
#     a.	Clockwise
#     b.	Anti-clockwise
#     c.	Repetition of input to output
# 9.	Complexity of colors : simple, medium & hard calculated  by (Sum) of color in set.
# ![image.png](attachment:image.png)
#  
# 
# 10.	If we can classify the data in above then its easy to solve the Simple & medium set.
# 
# 

# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:09:37 2020

@author: Sangram Phadke
"""
import numpy as np
import pandas as pd

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors

# for dirname, _, filenames in os.walk('Abstraction-and-Reasoning-Challenge'):
#     print(dirname)
    
training_path = Path('../input/abstraction-and-reasoning-challenge/')
SUBMISSION_PATH = Path('../input/abstraction-and-reasoning-challenge/')

training_path = training_path / 'training'
SUBMISSION_PATH = SUBMISSION_PATH / 'sample_submission.csv'

data_path = Path('training')
training_path = data_path


training_tasks = sorted(os.listdir(training_path))

list_training_tasks = []
list_training_tasks.append(training_tasks)

alist_training_tasks =np.array(list_training_tasks).reshape(400,1)
df = pd.DataFrame(data = alist_training_tasks)

#print(training_tasks[:3])

# task_file = str(training_path / '0520fde7.json')
# task_file = str(training_path / 'bd4472b8.json')
#df = pd.DataFrame(,columns='') 

list_training_tasks_number = []


list_train_pairs=[]


rows_train_input  = []


columns_train_input = []


task_file_name = []

colour_train_input = []


rows_train_output  = []


columns_train_output = [] 

colour_train_output = []

list_test_pairs=[]


rows_test_input = []

columns_test_input = []

colour_test_input = [] 

rows_test_output = []

columns_test_output = []

colour_test_output = [] 




for i in [0]:
    list_training_tasks_number.append(i)

    task_file = str(training_path / training_tasks[i])
   
    task_file_name.append(task_file)


    with open(task_file, 'r') as f:
        task = json.load(f)


    # task dictionary
    # print(task.keys())
    # print('\n') 
    
    n_train_pairs = len(task['train'])
    list_train_pairs.append(n_train_pairs)
    
    n_test_pairs = len(task['test'])
    list_test_pairs.append(n_test_pairs)
    
    #print(f'task contains {n_train_pairs} training pairs\n')
    #print(f'task contains {n_test_pairs} test pairs\n')


   
    
    for tp in range(0,n_train_pairs):
        
        #number of rows of each train input
        rtp = len(task['train'][tp]['input'])
        #print('Rows   of  train input {} of grid {} : '.format(tp,i),len(task['train'][tp]['input']))
        rows_train_input.append(rtp)
        
        #number of columns of each train input
        ctp = task['train'][tp]['input']
        actp = np.array(ctp).shape
        ##print('Columns of train input {} of grid {} : '.format(tp,i),(actp[1]))
        columns_train_input.append(actp[1])

        #number of color of each train input
        cotp = task['train'][tp]['input']
        #print('Colour of train input {} of grid {} : '.format(tp,i),task['train'][tp]['input'])
        colour_train_input.append(cotp)
        
        #number of rows of each train output
        rotp = len(task['train'][tp]['output'])
        #print('Rows of train output {} of grid {} : '.format(tp,i),len(task['train'][tp]['output']))
        rows_train_output.append(rotp)
        
        #number of columns of each train output
        cotp = task['train'][tp]['output']
        acotp = np.array(cotp).shape
        #print('Columns of train output {} of grid {} :'.format(tp,i),(acotp[1]))
        columns_train_output.append(acotp[1])
        
        #number of colors of each train output
        fcolro = task['train'][tp]['output']
        #print('Colour of train output {} of grid {} : '.format(tp,i),task['train'][tp]['output']) 
        colour_train_output.append(fcolro)

    for te in range(0,n_test_pairs):
        
        #number of rows of each test input
        frite = len(task['test'][te]['input'])
        #print('Rows of train input {} of grid {} : '.format(te,i),len(task['test'][te]['input']))
        rows_test_input.append(frite)
        
        #number of columns of each test input  
        fcite = task['test'][te]['input']
        afcite = np.array(fcite).shape
        #print('Columns of train input {} of grid {} :'.format(te,i),afcite[1])
        columns_test_input.append(afcite[1])
        
        #number of colours of each test input
        fcolrto = task['test'][te]['input']
        #print('Colour of train input {} of grid {} : '.format(te,i),task['test'][te]['input'])
        colour_test_input.append(fcolrto)
        
        #number of rows of each test output
        frote = len(task['test'][te]['output'])
        #print('Rows of train output {} of grid {} : '.format(te,i),len(task['test'][te]['input']))
        rows_test_output.append(frote)
        
        #number of columns of each test output
        fcote = task['test'][te]['output']
        afcote = np.array(fcote).shape
        #print('Columns of train input {} of grid {} :'.format(te,i),afcote[1])
        columns_test_output.append(afcote[1])
        
        #number of colours of each test output
        fcolrtoo = task['test'][te]['output']
        #print('Colour of train output {} of grid {} : '.format(te,i),task['test'][te]['input'])
        colour_test_output.append(fcolrtoo)
        


alist_training_tasks_number = np.array(list_training_tasks_number)
df0 = pd.DataFrame(list_training_tasks_number,columns=['list_training_tasks_number'])


alist_train_pairs = np.array(list_train_pairs)
df1 = pd.DataFrame(alist_train_pairs,columns=['list_train_pairs']) 

arows_train_input = np.array(rows_train_input)
df2 = pd.DataFrame(arows_train_input,columns=['rows_train_input']) 

acolumns_train_input = np.array(columns_train_input)
df3 = pd.DataFrame(acolumns_train_input,columns=['columns_train_input']) 

# atask_file_name = np.array(task_file_name)
# df4 = pd.DataFrame(task_file_name)

acolour_train_input = np.array(colour_train_input)
df5 = pd.DataFrame(colour_train_input)

arows_train_output = np.array(rows_train_output)
df6 = pd.DataFrame(arows_train_output,columns=['rows_train_output'])

acolumns_train_output = np.array(columns_train_output)
df7 = pd.DataFrame(acolumns_train_output,columns=['columns_train_output'])

acolour_train_output = np.array(colour_train_output)
df8 = pd.DataFrame(colour_train_output )

alist_test_pairs = np.array(list_test_pairs)
df9 = pd.DataFrame(alist_test_pairs ,columns=['list_test_pairs'])

arows_test_input = np.array(rows_test_input)
df10 = pd.DataFrame(arows_test_input,columns=['rows_test_input'])

acolumns_test_input = np.array(columns_test_input)
df11 = pd.DataFrame(acolumns_test_input,columns=['columns_test_input'])

acolour_test_input = np.array(colour_test_input)
df12 = pd.DataFrame(colour_test_input)

arows_test_output = np.array(rows_test_output)
df13 = pd.DataFrame(arows_test_output,columns=['rows_test_output'])

acolumns_test_output = np.array(columns_test_output)
df14 = pd.DataFrame(acolumns_test_output,columns=['columns_test_output'])

acolour_test_output = np.array(colour_test_output)
df15 = pd.DataFrame(colour_test_output)

df_all=pd.concat([df0,df1,df2,df3,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15],axis=1)


#df_all.to_csv('file1.csv',index=False)

#Un-comment to take print of file.



# #display the task
# def plot_task(task):
#     """
#     Plots the first train and test pairs of a specified task,
#     using same color scheme as the ARC app
#     """
#     cmap = colors.ListedColormap(
#         ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
#           '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
#     norm = colors.Normalize(vmin=0, vmax=9)
#     fig, axs = plt.subplots(1, 4, figsize=(15,15))
#     axs[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)
#     axs[0].axis('off')
#     axs[0].set_title('Train Input')
#     axs[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)
#     axs[1].axis('off')
#     axs[1].set_title('Train Output')
#     axs[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)
#     axs[2].axis('off')
#     axs[2].set_title('Test Input')
#     axs[3].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)
#     axs[3].axis('off')
#     axs[3].set_title('Test Output')
#     plt.tight_layout()
#     plt.show()
# plot_task(task)






# %% [markdown]
# ## Import libraries and define hyperparameters

# %% [code]
import os
import gc
import cv2
import json
import time

import numpy as np
import pandas as pd

from pathlib import Path
from keras.utils import to_categorical

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import colors

import plotly.figure_factory as ff

import torch
T = torch.Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# %% [code]
TEST_PATH = Path('../input/abstraction-and-reasoning-challenge/')
SUBMISSION_PATH = Path('../input/abstraction-and-reasoning-challenge/')

TEST_PATH = TEST_PATH / 'test'
SUBMISSION_PATH = SUBMISSION_PATH / 'sample_submission.csv'

# %% [markdown]
# ## Load the ARC data

# %% [markdown]
# ### Get testing tasks

# %% [code]
test_task_files = sorted(os.listdir(TEST_PATH))

test_tasks = []
for task_file in test_task_files:
    with open(str(TEST_PATH / task_file), 'r') as f:
        task = json.load(f)
        test_tasks.append(task)

# %% [markdown]
# ### Extract training and testing data

# %% [code]
Xs_test, Xs_train, ys_train = [], [], []

for task in test_tasks:
    X_test, X_train, y_train = [], [], []

    for pair in task["test"]:
        X_test.append(pair["input"])

    for pair in task["train"]:
        X_train.append(pair["input"])
        y_train.append(pair["output"])
    
    Xs_test.append(X_test)
    Xs_train.append(X_train)
    ys_train.append(y_train)

# %% [code] {"_kg_hide-input":true}
matrices = []
for X_test in Xs_test:
    for X in X_test:
        matrices.append(X)
        
values = []
for matrix in matrices:
    for row in matrix:
        for value in row:
            values.append(value)
            
Test_df = pd.DataFrame(values)
Test_df.columns = ["values"]

# %% [markdown]
# # Basic exploration

# %% [markdown]
# ## Look at a few train/test input/output pairs
# These are some of the pairs present in the training data. 

# %% [code] {"_kg_hide-input":true}
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
training_tasks = sorted(os.listdir(training_path))

for i in [1, 19, 8, 15, 9]:

    task_file = str(training_path / training_tasks[i])

    with open(task_file, 'r') as f:
        task = json.load(f)

    def plot_task(task):
        """
        Plots the first train and test pairs of a specified task,
        using same color scheme as the ARC app
        """
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        fig, ax = plt.subplots(1, 4, figsize=(15,15))
        ax[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)
        width = np.shape(task['train'][0]['input'])[1]
        height = np.shape(task['train'][0]['input'])[0]
        ax[0].set_xticks(np.arange(0,width))
        ax[0].set_yticks(np.arange(0,height))
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].tick_params(length=0)
        ax[0].grid(True)
        ax[0].set_title('Train Input')
        ax[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)
        width = np.shape(task['train'][0]['output'])[1]
        height = np.shape(task['train'][0]['output'])[0]
        ax[1].set_xticks(np.arange(0,width))
        ax[1].set_yticks(np.arange(0,height))
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].tick_params(length=0)
        ax[1].grid(True)
        ax[1].set_title('Train Output')
        ax[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)
        width = np.shape(task['test'][0]['input'])[1]
        height = np.shape(task['test'][0]['input'])[0]
        ax[2].set_xticks(np.arange(0,width))
        ax[2].set_yticks(np.arange(0,height))
        ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        ax[2].tick_params(length=0)
        ax[2].grid(True)
        ax[2].set_title('Test Input')
        ax[3].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)
        width = np.shape(task['test'][0]['output'])[1]
        height = np.shape(task['test'][0]['output'])[0]
        ax[3].set_xticks(np.arange(0,width))
        ax[3].set_yticks(np.arange(0,height))
        ax[3].set_xticklabels([])
        ax[3].set_yticklabels([])
        ax[3].tick_params(length=0)
        ax[3].grid(True)
        ax[3].set_title('Test Output')
        plt.tight_layout()
        plt.show()

    plot_task(task)

# %% [markdown]
# ## Number frequency <a id="number-frequency"></a>

# %% [code] {"_kg_hide-input":true}
px.histogram(df, x="values", title="Numbers present in matrices")

# %% [markdown]
# From the above graph, we can clearly see that the number distribution has a string positive skew. Most numbers in the matrices are clearly 0. This is reflected by the dominance of black color in most matrices.

# %% [markdown]
# ## Matrix mean values <a id="matrix-mean-values"></a>

# %% [code] {"_kg_hide-input":true}
means = [np.mean(X) for X in matrices]
fig = ff.create_distplot([means], group_labels=["Means"], colors=["green"])
fig.update_layout(title_text="Distribution of matrix mean values")

# %% [markdown]
# From the above graph, we can see that lower means are more common than higher means. The graph, once again, has a strong positive skew. This is further proof that black is the most dominant color in the matrices.

# %% [markdown]
# ## Matrix heights <a id="matrix-heights"></a>

# %% [code] {"_kg_hide-input":true}
heights = [np.shape(matrix)[0] for matrix in matrices]
widths = [np.shape(matrix)[1] for matrix in matrices]

# %% [code] {"_kg_hide-input":true}
fig = ff.create_distplot([heights], group_labels=["Height"], colors=["magenta"])
fig.update_layout(title_text="Distribution of matrix heights")

# %% [markdown]
# From the above graph, we can see that matrix heights have a much more uniform distribution (with significantly less skew). The distribution is somewhat normal with a mean of approximately 15.

# %% [markdown]
# ## Matrix widths <a id="matrix-widths"></a>

# %% [code] {"_kg_hide-input":true}
fig = ff.create_distplot([widths], group_labels=["Width"], colors=["red"])
fig.update_layout(title_text="Distribution of matrix widths")

# %% [markdown]
# From the above graph, we can see that matrix widths also have a uniform distribution (with significantly less skew). The distribution is also somewhat uniform with a mean of approximately 16.

# %% [markdown]
# ## Height vs. Width <a id="height-vs-width"></a>

# %% [code] {"_kg_hide-input":true}
plot = sns.jointplot(widths, heights, kind="kde", color="blueviolet")
plot.set_axis_labels("Width", "Height", fontsize=14)
plt.show(plot)

# %% [code] {"_kg_hide-input":true}
plot = sns.jointplot(widths, heights, kind="reg", color="blueviolet")
plot.set_axis_labels("Width", "Height", fontsize=14)
plt.show(plot)

# %% [markdown]
# From the above graphs, we can see that heights and widths have a strong positive correlation, *i.e.* greater widths generally result in greater heights. This is consistent with the fact that most matrices are square-shaped.

# %% [markdown]
# ### Define function to flatten submission matrices

# %% [code]
#Submission 

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# %% [markdown]
# ### Prepare submission dataframe

# %% [code]
test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]

for idx, pred in enumerate(test_predictions):
    test_predictions[idx] = flattener(pred)
    
submission = pd.read_csv(SUBMISSION_PATH)
submission["output"] = test_predictions

# %% [markdown]
# ### Convert submission to .csv format

# %% [code]
submission.head()

# %% [code]
submission.to_csv("submission.csv", index=False)

# %% [markdown]
# # Ending note:
# 
# From the other Notebooks Copyed the code to read the data in my note book. rest of the core code & analysis is my own (Feel free to do same with this note book if you find valuable)
# 
# Imp Note: Big thanks to  Walter & Tarun Paparaju many more..
# 
# Pease upvote if like it.
