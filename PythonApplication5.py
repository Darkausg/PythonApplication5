import pandas as pd
import os


"""

Lecteur des csv

"""
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



def read_file_from_csv(file_path):
    try:
        list_output = pd.read_csv(file_path,sep=",",header=None,skipinitialspace=True,quotechar='"').values.tolist()
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        exit()
    except FileNotFoundError:
        print(f"Le fichier {file_path} est introuvable.")
        exit()
    return list_output

def send_simple_list_to_txt(lst,file_name):
    try:
        with open("simple_liste" + file_name + ".txt", 'w') as f:
            for i in lst:
                f.write( i + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
    pass

def read_simple_list_from_txt(file_name):
    try:
        with open("simple_liste" + file_name + ".txt", 'r') as f:
            content = f.readlines()
        # Supprime les caractères de fin de ligne (\n)
        return [line.strip() for line in content]
    except FileNotFoundError:
        print(f"Le fichier {file_name} est introuvable.")
        return []
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return []
    pass

def send_list_of_list_to_csv(lst,file_name):
    try:
        with open("list_of_list" + file_name + ".csv","w") as f:
            for i in lst:
                f.write(i[0])
                for j in range(1,len(i)):
                    f.write(", " + i[j] )
                f.write("\n")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    return 0



def send_dico_to_csv(dic,file_name):
    try:
        with open("dico" + file_name + ".csv","w",encoding="utf-8",newline="") as f:
            for i in dic:
                f.write(i + "," + str(dic[i]) + "\n")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    pass

def read_dico_from_csv(file_name):
    dico={}
    try:
        with open("dico" + file_name + ".csv", 'r',encoding="utf-8") as f:
            content = f.readlines()
        # Supprime les caractères de fin de ligne (\n)
        lst = [line.strip() for line in content]
        #on extrait le dico
        for i in lst:
            dico[i.split(",")[0]] = int(i.split(",")[1])#cette conversion est possible car toutes les valeurs dans dico sont des entiers, du moins dans le cadre de ce TP 
            pass
        return dico
    except FileNotFoundError:
        print(f"Le fichier {file_name} est introuvable.")
        return []
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return []
    pass

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

#emoji = 2 cara speciaux diferents; ou  les prendres de la liste envoyé pr mail

#generer par Chatgpt
#il faudra nettoyer cette partie
emojis_clavier = [
    # Visages heureux
    ":)", ":-)", ":D", ":-D", "8D", "8-D", "xD", "XD", "=D", "=)",  
    "^-^", "^^", "^_^", "n_n", "U_U", "UwU", "uwu", "OwO", "owo",  
    "(^_^)", "(o^_^o)", "(n_n)", "(^-^)", "(=^_^=)",  
   
    # Visages tristes et neutres
    ":(", ":-(", ":'(", ":'-(", "T_T", "TT_TT", "T-T", "-_-",  
    "v_v", "u_u", "._.", "-.-", "(._.)", "(>_<)", "(;-;)",  
   
    # Surprise et choc
    ":O", ":-O", ":o", ":-o", "O_O", "o_O", "O_o", "o_o", "(o_o)", "0_0",  
   
    # Clins d'œil et taquineries
    ";)", ";-)", ";P", ";-P", ";D", ";-D",  
   
    # Expressions diverses
    ":P", ":-P", "XP", "X-P", "xP", "x-P", ":/", ":-/", ":|", ":-|",  
    "B)", "B-)",  "(?_?)", "(>_>)", "(<_<)"
]


test = [[x for x in row.split() ] for row in a]
#on parcourt chaque ligne de test
#   1) on retire les emojis
#   2) avoir une fonction qui affiche tout mots avec caractere speciaux qui nesont pas des emoji pour s'assurer qu'on a tout les mots
#   3) on retire les caracteres speciaux, on mets tout en minuscule yadi-yada
send_list_of_list_to_csv(test,"essais")
ar = 0


