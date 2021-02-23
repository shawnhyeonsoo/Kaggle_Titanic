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
    if len(prefix_dict[p]) > 2:
        prefix_average[p] = int(sum(sorted(prefix_dict[p][1:-1]))//len(prefix_dict[p][1:-1]))
    else:
        prefix_average[p] = int(sum(sorted(prefix_dict[p])) // len(prefix_dict[p]))


#################################
######MAKE TRAIN FILE############
#################################

f = pd.read_csv('train.csv')
f = pd.get_dummies(data = f, columns = ['Survived','Pclass','Sex'])
#f2 = open('train.csv', 'r',encoding = 'utf8').readlines()

#Head: 'PassengerId, Survived, 'PClass, Name, Sex, Age, SibSP, Parch, Ticket, Fare, Cabin, Embarked'
t = open('train_boundaries.txt','w',encoding = 'utf8')
train_dataset = []
for i in range(0,len(f)):
    line = list(f.iloc[i])
    age = [0 for i in range(9)]
    sibsp = [0 for i in range(9)]
    parch = [0 for i in range(10)]
    fare = [0 for i in range(6)]

    if str(f.iloc[i]['Age']) == 'nan':
        preff = re.search(' [A-Za-z]+\.', f.iloc[i]['Name']).group()
        predicted_age = prefix_average[preff]
        pos = predicted_age//10
        age[pos] = 1
        sibsp[f.iloc[i]['SibSp']] = 1
        parch[f.iloc[i]['Parch']] = 1
        fare_pos = int(f.iloc[i]['Fare'])//100
        fare[fare_pos] = 1
        extracted = line[11:] + age + sibsp + parch + fare

    else:
        pos = int(f.iloc[i]['Age'])//10
        age[pos] = 1
        sibsp[f.iloc[i]['SibSp']] = 1
        parch[f.iloc[i]['Parch']] = 1
        fare_pos = int(f.iloc[i]['Fare'])//100
        fare[fare_pos] = 1
        extracted = line[11:] + age + sibsp + parch + fare

    t.writelines(",".join(map(str,extracted)) + '\n')
t.close()



###Make Test Boundary file ###
f2 = pd.read_csv('test.csv')
f2 = pd.get_dummies(data = f2, columns = ['Pclass','Sex'])

t2 = open('test_boundaries.txt','w',encoding = 'utf8')
train_dataset = []
for t in range(0,len(f2)):
    line = list(f2.iloc[t])
    age = [0 for i in range(9)]
    sibsp = [0 for i in range(9)]
    parch = [0 for i in range(10)]
    fare = [0 for i in range(6)]
    if str(f2.iloc[t]['Age']) == 'nan':
        preff = re.search(' [A-Za-z]+\.', f2.iloc[t]['Name']).group()
        predicted_age = prefix_average[preff]
        pos = predicted_age//10
    else:
        pos = int(f2.iloc[t]['Age'])//10
        age[pos] = 1

    if str(f2.iloc[t]['Fare']) == 'nan':
        fare_pos = 50//100
    else:
        fare_pos = int(f2.iloc[t]['Fare']) // 100

    age[pos] = 1
    sibsp[f2.iloc[t]['SibSp']] = 1
    parch[f2.iloc[t]['Parch']] = 1
    fare[fare_pos] = 1
    extracted2 = line[9:] + age + sibsp + parch + fare


    t2.writelines(",".join(map(str,extracted2)) + '\n')
t2.close()
