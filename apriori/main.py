from apyori import apriori
import pandas as pd

data = pd.read_csv('sepet.csv',header=None)

t_list = []

for i in range(0,7501):
    t_list.append([str(data.values[i,j])for j in range(0,20)])

rules = apriori(t_list,min_support=0.01,min_confidence=0.2,min_lift=2)
print(list(rules))