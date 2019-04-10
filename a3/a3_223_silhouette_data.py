import pickle
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
from collections import defaultdict


df = pd.DataFrame()

for alg in ['kmeans', 'mog']:
    for k in [5,10,15,20,30]:
        df.loc[k, 'k'] = k
        for i in range(10):
            filename = '2.2.3-silhouette-%s/data-%s-%s.pkl' % (i, alg, k)
            print(filename)
            data = pickle.load(open(filename, 'rb'))
            X = data['train']['X']
            y = data['train']['y']
            score = silhouette_score(X, y)
            df.loc[k, '%s_%s' % (alg, i)] = score
    
    df['%s_mean' % alg] = df[['%s_%s' % (alg, i) for i in range(10)]].mean(axis=1)
    df['%s_std' % alg] = df[['%s_%s' % (alg, i) for i in range(10)]].std(axis=1)

df.reset_index(drop=True, inplace=True)
df.to_csv('silhouette_score.csv')