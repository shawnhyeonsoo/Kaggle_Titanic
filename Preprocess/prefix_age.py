import pandas as pd
import csv
import re
from collections import Counter
f = pd.read_csv('train.csv')
f = pd.get_dummies(data = f, columns = ['Survived','Pclass','Sex'])
name_list = []
for i in list(f['Name']):
    name_list.append(re.search(' [A-Za-z]+\.', i).group())

prefix_dict = {}
prefix_average = {}
prefix = list(set(name_list))
for j in prefix:
    prefix_dict[j] = []
    prefix_average[j] = set()

for k in range(len(f)):
    pref = re.search(' [A-Za-z]+\.',f.iloc[k]['Name']).group()
    if str(f.iloc[k]['Age']) == 'nan':
        pass
    else:
        prefix_dict[pref].append(f.iloc[k]['Age'])

for p in prefix_dict:
    prefix_average[p] = int(sum(prefix_dict[p])//len(prefix_dict[p]))
#prefixes = Counter(name_list).most_common()
