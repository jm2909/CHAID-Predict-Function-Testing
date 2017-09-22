import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Independent_data = pd.read_csv("Independent_data.csv")
Response = pd.read_csv("Response.csv")
X_train, X_test, y_train, y_test = train_test_split(Independent_data,Response,test_size = 0.25,random_state = 42)

train_data = pd.concat([X_train,y_train],axis = 1)
test_data = pd.concat([X_test,y_test],axis = 1)

from CHAID import Tree
tree = Tree.from_pandas_df(test_data, dict(zip(X_train.columns.tolist(), list(np.repeat('nominal',len(X_train.columns))))),
                           y_train.columns[0],max_depth=10,min_child_node_size = 10)
tree.print_tree()
tree.classification_rules()




def predict(df,tree):
    rules = tree.classification_rules()
    lenrules  = len(rules)
    j=0
    df.index = range(0,df.shape[0])
    Response = np.repeat(0, df.shape[0])
    while(j <= lenrules-1):
        r1 = rules[j]
        ruleset = r1.items()[1][1]
        lenruleset = len(ruleset)
        k = 0
        df1 = df
        while(k <= lenruleset-1):
            r = ruleset[k]
            v = r.get('variable')
            d = r.get('data')
            ind = []
            for i in range(0,df1[v].shape[0]):
                if df1[v].iloc[i] in d:
                    ind.append(df1.index[i])
            df1 = df1.loc[ind]
            if k == lenruleset-1:
                xset = tree.tree_store[r1.items()[0][1]]._members
                perc0,perc1 = xset.get(0),xset.get(1)
                print("Node:",j,perc0,perc1)
                if perc0 >= perc1:
                    Response[ind] =0
                else:
                    Response[ind] = 1
            k = k+1
        j = j+1
    return Response

PredictedClass = predict(test_data,tree)
