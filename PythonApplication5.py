"""

Lecteur des csv

"""

import pandas as pd

train_df = pd.read_csv("./tweets_train.csv", sep=",", header=None, skipinitialspace=True, quotechar='"').values.tolist()
dev_df = pd.read_csv("./tweets_dev.csv", sep=",", header=None, skipinitialspace=True, quotechar='"').values.tolist()
test_df = pd.read_csv("./tweets_test.csv", sep=",", header=None, skipinitialspace=True, quotechar='"').values.tolist()
"""
print(train_df)
print(dev_df)
print(test_df)
"""

"""
Projet BE
"""

#print(train_df)

def seperate_pos_nega(liste_tweet):
    train_pos = []
    train_neg =[]
    for i in liste_tweet:
        if i[0] =="positive":
            train_pos.append(i[1])
        else:
            train_neg.append(i[1])
    return train_pos,train_neg

a,b = seperate_pos_nega(train_df)
c=1
print(a)

def sep_tweet_label(liste_tweet):
    label = []
    tweet = []
    for i in liste_tweet:
        label = i[0]
        tweet = i[1]
    return label,tweet

#emoji = 2 cara speciaux diferents