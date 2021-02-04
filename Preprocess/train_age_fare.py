import pandas as pd
import re

f = pd.read_csv('train.csv')
#f = f['Embarked'].fillna('S')
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

#################################
######MAKE TRAIN FILE############
#################################

f = pd.read_csv('train.csv')
f = pd.get_dummies(data = f, columns = ['Survived','Pclass','Sex'])
#f2 = open('train.csv', 'r',encoding = 'utf8').readlines()

#Head: 'PassengerId, Survived, 'PClass, Name, Sex, Age, SibSP, Parch, Ticket, Fare, Cabin, Embarked'
t = open('train_fare.txt','w',encoding = 'utf8')
train_dataset = []
for i in range(0,len(f)):
    line = list(f.iloc[i])
    if str(f.iloc[i]['Age']) == 'nan':
        preff = re.search(' [A-Za-z]+\.', f.iloc[i]['Name']).group()
        extracted = line[11:] + [int(o) for o in '{0:010b}'.format(prefix_average[preff])] + [int(p) for p in '{0:010b}'.format(int(line[3]))] + [int(q) for q in '{0:010b}'.format(int(line[4]))] + [int(m) for m in '{0:012b}'.format(int(f.iloc[i]['Fare']))]
    else:
        extracted = line[11:] + [int(o) for o in '{0:010b}'.format(int(line[2]))] + [int(p) for p in '{0:010b}'.format(int(line[3]))] + [int(q) for q in '{0:010b}'.format(int(line[4]))]+ [int(m) for m in '{0:012b}'.format(int(f.iloc[i]['Fare']))]
    t.writelines(",".join(map(str,extracted)) + '\n')

t.close()
