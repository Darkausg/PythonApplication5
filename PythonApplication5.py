from time import sleep
import pandas as pd
import os



class Proba:
    """
    Args:
        p_m_sachant_pos      #proba de chaque mot sachant que le label est positif : P(m|POS)
        p_m_sachant_neg      #proba de chaque mot sachant que le label est negatif : P(m|NEG)
        p_m                  #proba du mot dans le corpus                            P(m)
        p_pos                #proba tweet positif parmi ensemble de tweet            P(POS)
        p_neg                #proba tweet negatif parmi ensemble de tweet            P(NEG)
    """
    def __init__(self,p_m_sachant_pos,p_m_sachant_neg,p_m,p_pos,p_neg):
        self.p_m_sachant_pos = p_m_sachant_pos      #proba de chaque mot sachant que le label est positif : P(m|POS)
        self.p_m_sachant_neg = p_m_sachant_neg      #proba de chaque mot sachant que le label est negatif : P(m|NEG)
        self.p_m = p_m                      #proba du mot dans le corpus                            P(m)
        self.p_pos = p_pos                        #proba tweet positif parmi ensemble de tweet            P(POS)
        self.p_neg = p_neg                        #proba tweet negatif parmi ensemble de tweet            P(NEG)
    
"""

Lecteur des csv

"""

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
        with open("simple_liste" + file_name + ".txt", 'w',encoding="UTF-8") as f:
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
        with open("list_of_list" + file_name + ".csv","w",encoding="utf-8") as f:
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
        with open("dico" + file_name + ".csv","w",newline="") as f:
            for i in dic:
                f.write(i + "," + str(dic[i]) + "\n")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    pass

def read_dico_from_csv(file_name):
    dico={}
    try:
        with open("dico" + file_name + ".csv", 'r') as f:
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




def sep_tweet_label(liste_tweet):
    label = []
    tweet = []
    for i in liste_tweet:
        label.append(i[0])
        tweet.append(i[1])
    return label,tweet

#emoji = 2 cara speciaux diferents; ou  les prendres de la liste envoyé pr mail

#generer par Chatgpt
#il faudra nettoyer cette partie
emojis_clavier = [
    # Visages heureux
    ":)", ":-)", ":D", ":-D", "8D", "8-D", "xD", "XD", "=D", "=)",  
    "^-^", "^^", "^_^", "n_n", "U_U", "UwU", "uwu", "OwO", "owo",  
    "(^_^)", "(o^_^o)", "(n_n)", "(^-^)", "(=^_^=)",  "<3", ":')" , ":'(" ,
   
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

emoji_lower = [x.lower for x in emojis_clavier]

#test = [[x for x in row.split() ] for row in pos]
#on parcourt chaque ligne de test
#   1) on retire les emojis
#   2) avoir une fonction qui affiche tout mots avec caractere speciaux qui nesont pas des emoji pour s'assurer qu'on a tout les mots
#   3) on retire les caracteres speciaux, on mets tout en minuscule yadi-yada
print("\n\n\n\n")
#print(pos[329])
#send_list_of_list_to_csv(test,"essais")
ar=10




def liste_occurrences(l):
    """
    prend une liste de mots l en entree et compte le nombre de fois que chaque mot apparait dans cette liste. 
    Elle retourne un dictionnaire res ou les cles sont les mots et les valeurs sont les occurrences de ces mots.
    """
    res=dict()
    for i in l:
        if not i in res:
            res[i]=1
        else:
            res[i]+=1
    return res



def reduction_occurence_unique(d):
    """
    prend en entree un dictionnaire (d) ou les cles sont des mots et les valeurs sont le nombre d'occurrences de chaque mot. 
    retourne un nouveau dictionnaire (res) qui ne contient que les mots apparaissant plus d'une fois dans le dictionnaire d'origine.
    """
    res=dict()
    for i in d:
        if not d[i]==1:
            res[i]=d[i]
    return res



def tri_dico(d,ascending):
    dico_trie = dict(sorted(d.items(), key=lambda item: item[1],reverse=ascending))
    return dico_trie

def only_caracter(ori_text):
    if ((ori_text in emoji_lower) or (ori_text in emojis_clavier)):
        return ori_text
    if ori_text.isalpha():#que des lettres
        return ori_text
    if ori_text.isalnum():#que des numbres
        pass
    if (ori_text in function_words):
        pass
    letter ="azertyuiopqsdfghjklmwxcvbn"
    text_treated = ""
    for i in ori_text:
        if ( i in letter ) :
            text_treated = text_treated + i
        else:
            text_treated = text_treated + "SCP"
    #il faudra retirer les espaces
            """
    if (text_treated in [" "*k for k in range(2,5)]):
        if (ori_text in emoji_lower):
            print("GIGA PROBLEME")
            sleep(1)
        print("\n\n On a un pb")
        print(ori_text)
        print("1" + text_treated + "1")
        sleep(0.25)
        """

    return text_treated


def reduction_occurence(d,t):
    if len(d)<t:
        return d
    for i in range(t):
        d.popitem()
    return d

def conca_str_in_list(lst):
    conca_lst = " ".join(lst)
    return conca_lst

def subtraitement(string):
    """
    revoie une liste de mots

    Args:
        string (str)

    Return:
        word_list (str)
    """
    temp=string.lower()
    temp2= temp.split()#temp2 est une liste
    send_simple_list_to_txt(temp2,"temp2")
    #supprimez cara speciaux et garder smiley
    #supprimer tout ce qui commence par https
    a=10
    list_word2 = [only_caracter(x)  for x in temp2]
    send_simple_list_to_txt(list_word2,"list_word2")
    space = [" " * z for z in range(15)]
    list_word = [k for k in list_word2 if (not k in space)]
    words_list = "SCP".join(list_word)
    send_simple_list_to_txt(words_list.split("SCP"),"WORKNOW")
    return (words_list.split("SCP"))

def traitement1(lst):

    words_list = subtraitement(lst)

    # Create initial word occurrence dictionary
    initial_word_dict=liste_occurrences(words_list)

    # Remove words with a single occurrence
    #unique_occurences_dict=reduction_occurence_unique(initial_word_dict)
    unique_occurences_dict = initial_word_dict

    # Sort the dictionary in descending order of occurrences
    sorted_dict_desc=tri_dico(unique_occurences_dict,False)
    
    # Reduce the dictionary to a specific number of occurrences
    reduced_dict=reduction_occurence(sorted_dict_desc,0)#redu_ocu, changer 100 en un truc réglable

    # Sort the dictionary in ascending order of occurrences
    final_sorted_dict=tri_dico(reduced_dict,True)

    return final_sorted_dict,words_list

#liste des stop-words (les mots les plus fréquents qui ne portent pas d'information), oui c'est généré par IA
function_words = [
    "virginamerica"
    # Articles
    "a", "an", "the",
    
    # Conjonctions
    "and", "but", "or", "nor", "for", "yet", "so", "although", "because", "if", "unless", "while", "since",
    
    # Prépositions
    "in", "on", "at", "by", "with", "about", "of", "for", "to", "from", "over", "under", "between", 
    "through", "against", "into", "onto", "upon", "out", "off", "along", "inside", "outside", "beneath",
    
    # Pronoms personnels
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    
    # Pronoms possessifs
    "mine", "yours", "his", "hers", "ours", "theirs",
    
    # Pronoms interrogatifs
    "who", "whom", "whose", "which", "what",
    
    # Adverbes fonctionnels
    "not", "very", "just", "then", "there", "here", "well", "so", "now", "already", "still",
    
    # Modaux (verbes auxiliaires)
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    
    # Déterminants
    "this", "that", "these", "those", "each", "every", "some", "any", "no", "all", "both", "either", "neither",
    
    # Particules
    "to", "up", "out", "off", "down",
    
    # Quantifieurs
    "few", "many", "much", "several", "most", "none",
    
    # Autres mots fonctionnels
    "do", "does", "did", "been", "being", "has", "have", "had", "am", "is", "are", "was", "were", "be"
]




#envoyer les dico et liste en CSV

def p_mot(dico,effectif):
    res = {}
    for i in dico:
        res[i] =  dico[i]/effectif
    return res

def classification_tweet(string,data_proba):#lst un string (dans ce cas il s'agit d'un tweet)

    p_prediction_pos = data_proba.p_pos  #var indiquant la proba que le tweet est positif sachant les mots du tweets
    p_prediction_neg = data_proba.p_neg  #var indiquant la proba que le tweet est negatif sachant les mots du tweets
    #traitement texte
    tweet = subtraitement(string)
    #calcul proba
    for i in tweet: #i est un mot du tweet
        #demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
        if (i in data_proba.p_m_sachant_pos):
            p_prediction_pos = p_prediction_pos * ( data_proba.p_m_sachant_pos[ i ] / data_proba.p_m[i] )
        if (i in data_proba.p_m_sachant_neg):
            p_prediction_neg = p_prediction_neg * ( data_proba.p_m_sachant_neg[i] / data_proba.p_m[i] )
    if (p_prediction_pos > p_prediction_neg) :        #proba pos > prob neg
        return "positive"
    else:           #proba neg > proba pos
        return "negative"
    pass

#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
#demander au prof comment on fait quand un mot n'est pas dans le dico des mots positif ou negatif
def main():
    if (os.path.exists("dicodico_essais_train_pos.csv") and os.path.exists("dicodico_essais_train_neg.csv") and os.path.exists("dicodico_essais_corp.csv")):
        train_df = pd.read_csv("./tweets_train.csv", sep=",", header=None, skipinitialspace=True, quotechar='"').values.tolist()
        dev_df = pd.read_csv("./tweets_dev.csv", sep=",", header=None, skipinitialspace=True, quotechar='"').values.tolist()
        test_df = pd.read_csv("./tweets_test.csv", sep=",", header=None, skipinitialspace=True, quotechar='"').values.tolist()
        pos,neg = seperate_pos_nega(train_df)
        #print(pos)
        c=1
        conca_pos = conca_str_in_list(pos)
        conca_neg = conca_str_in_list(neg)
        pos_final_sorted_dict,pos_words_list = traitement1(conca_pos)
        neg_final_sorted_dict,neg_words_list = traitement1(conca_neg)
        #print(pos_final_sorted_dict)

        corp_tot_final_sorted_dict,corp_tot_words_list = traitement1(conca_pos+" 1 " + conca_neg)

        send_dico_to_csv(pos_final_sorted_dict,"dico_essais_train_pos")
        send_dico_to_csv(neg_final_sorted_dict,"dico_essais_train_neg")
        send_dico_to_csv(corp_tot_final_sorted_dict,"dico_essais_corp")

    else:
        
        zr=0
    a=120
    p_pos = len(pos)/(len(pos)+len(neg))#proba tweet positif parmi ensemble de tweet
    p_neg = len(neg)/(len(pos)+len(neg))#proba tweet negatif parmi ensemble de tweet
    n_corp = (len(pos_words_list)+len(neg_words_list))#nombre total de mot dans le corpus
    p_mot_dic_pos = p_mot(pos_final_sorted_dict,len(pos_words_list))#proba de chaque mot sachant que le label est positif
    p_mot_dic_neg = p_mot(neg_final_sorted_dict,len(neg_words_list))#proba de chaque mot sachant que le label est negatif
    clear = "\n"*50
    """print(clear)
    print(p_pos)
    print(p_neg)
    print(p_pos+p_neg)
    print("nbr omt")
    print(n_corp)"""
    p_mot_dic=p_mot(corp_tot_final_sorted_dict,n_corp)#proba du mot dans le corpus

    proba_train = Proba(p_mot_dic_pos,p_mot_dic_neg,p_mot_dic,p_pos,p_neg)#on rassemble toutes les variables dans un seul objet pour se simplifier la vie
    #print(dev_df)
    
    label_dev,tweet_dev=sep_tweet_label(dev_df)
    prediction = [classification_tweet(x,proba_train) for x in tweet_dev]
    print(prediction)
    nbr_pos = prediction.count("positive")
    nbr_neg = prediction.count("negative")
    print(f"L'element positive apparait {nbr_pos} fois dans la liste.")
    print(f"L'element negative apparait {nbr_neg} fois dans la liste.")
    pourcent = nbr_pos/len(prediction)
    print(f"il y a {pourcent}% d'element positif")
    pourcent = pourcent+0.2228
    print(pourcent)
    """
    label_dev,tweet_dev=sep_tweet_label(test_df)
    prediction = [classification_tweet(x,proba_train) for x in tweet_dev]
    print(prediction)
    nbr_pos = prediction.count("positive")
    nbr_neg = prediction.count("negative")
    print(f"L'element positive apparait {nbr_pos} fois dans la liste.")
    print(f"L'element negative apparait {nbr_neg} fois dans la liste.")
    pourcent = nbr_pos/len(prediction)
    print(f"il y a {pourcent}% d'element positif")
    """
    #on trouve 22% de positif et on a 90% de reussite quand on compare au label qui existe
    compare = []
    for i in range(len(prediction)):
        if (prediction[i]==label_dev[i]):
            compare.append(1)
        else:
            compare.append(0)
    accuracy = compare.count(1)/len(compare)
    print("accuracy est de :")
    print(accuracy)
    rzse=0
    pass

main()
#top pour projet 94%

