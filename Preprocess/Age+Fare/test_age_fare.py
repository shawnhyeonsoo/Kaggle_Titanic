import pandas as pd
import csv
f = pd.read_csv('test.csv')
f = f.fillna(0)
f = pd.get_dummies(data = f, columns = ['Pclass','Sex'])
#f2 = open('test.csv', 'r',encoding = 'utf8').readlines()

#Head: 'PassengerId, Survived, 'PClass, Name, Sex, Age, SibSP, Parch, Ticket, Fare, Cabin, Embarked'
t = open('test_age_fare.txt','w',encoding = 'utf8')
train_dataset = []
for i in range(0,len(f)):
    line = list(f.iloc[i])
    if str(f.iloc[i]['Fare']) == 'nan':
        extracted = line[9:] + [int(o) for o in '{0:010b}'.format(int(line[2]))] + [int(p) for p in '{0:010b}'.format(int(line[3]))] + [int(q) for q in '{0:010b}'.format(int(line[4]))] + [int(m) for m in '{0:012b}'.format(15)]
    else:
        extracted = line[9:] + [int(o) for o in '{0:010b}'.format(int(line[2]))] + [int(p) for p in '{0:010b}'.format(int(line[3]))] + [int(q) for q in '{0:010b}'.format(int(line[4]))]+ [int(m) for m in '{0:012b}'.format(int(f.iloc[i]['Fare']))]
    t.writelines(",".join(map(str,extracted)) + '\n')

t.close()
