
# coding: utf-8

# # Extract subset

# In[3]:


import subprocess
import pandas as pd

dataDir = 'data'
inputPedMap = 'data/HACK_15_one'

assoc = pd.read_csv('data/Marta-ped_sorted.assoc', sep=' ')

extractNums = [250, 500, 750, 1000]
# extractNums = [70, 100, 200, 250, 500, 750, 1000]

for extractNum in extractNums:

    snpsPath = dataDir+'/SNPs'+str(extractNum)+'.txt'

    assoc['SNP'][:extractNum].to_csv(snpsPath, sep=' ', index=False)  

    outPath = dataDir+'/biohack_'+str(extractNum)
    # --allele-1234
    plinkCmd = 'plink2/plink --file '+inputPedMap+' --allele-1234 --extract '+snpsPath+' --recode --out '+outPath

    print(plinkCmd)

    subprocess.run(plinkCmd , shell=True, check=True)



# # Prepare Data

# In[4]:



for extractNum in extractNums:
    with open(dataDir+'/biohack_'+str(extractNum)+'.ped') as file:
        with open(dataDir+'/biohack_'+str(extractNum)+'_AA_no.ped', 'w') as writeF:
            for nom in range(5901):
                line = file.readline()
                line = line.replace('I', '5')
                line = line.replace('D', '6')
                line = line.split(' ')

                if line[5] == '-9':
                    continue
                

                arr = []
                for num in range(6):
                    arr.append(line[num])


                for i in range(6,len(line), 2):
                    arr.append(line[i] + line[i+1]) 


                writeF.write(' '.join(arr))
            


# # Prediction
# Zastosowanie metod uczenia nadzorowanego do przewidywania koloru oczu. 

# In[5]:


import pandas as pd
import numpy as np


# ## Prepare training and testing data

# In[6]:


results = pd.DataFrame(columns=['Num_of_rs', 'algo', 'score', 'params'])
results


# In[7]:


importances = {}


# In[44]:


from sklearn.model_selection import train_test_split

extractNum = 750

inPath = 'data/biohack_'+str(extractNum)+'_AA_no.ped'

pedData = pd.read_csv(inPath, sep=' ', names=range(75))

x_train, x_test, y_train, y_test = train_test_split(pedData.iloc[:,5:], pedData.iloc[:,4], test_size = 0.1)


# In[50]:


len(x_train)


# ## SVM

# In[45]:


from sklearn import svm

params = 'default'

clf = svm.SVC()
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)

df = pd.DataFrame([[extractNum, 'SVM', score, params]], columns=['Num_of_rs', 'algo', 'score', 'params'])
results = results.append(df, ignore_index=True)
results
# df


# ## Neural Network - MLPClassifier

# In[46]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs')
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)

df = pd.DataFrame([[extractNum, 'MLP', score, params]], columns=['Num_of_rs', 'algo', 'score', 'params'])
results = results.append(df, ignore_index=True)
results


# ## Stochastic Gradient Descent

# In[47]:


from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)

df = pd.DataFrame([[extractNum, 'SGD', score, params]], columns=['Num_of_rs', 'algo', 'score', 'params'])
results = results.append(df, ignore_index=True)
results


# ## DecisionTreeClassifier

# In[48]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)

df = pd.DataFrame([[extractNum, 'Tree', score, params]], columns=['Num_of_rs', 'algo', 'score', 'params'])
results = results.append(df, ignore_index=True)
results

# 


# In[37]:


results.to_csv('results2.csv', sep='\t')


# In[54]:


importances = {extractNum: clf.feature_importances_}
# importances = np.append(importances, clf.feature_importances_)
importances

