import re
import pandas as pd
import math
import os
from typing import Dict
from ast import Dict, Pass
from time import sleep




class Data:
    def __init__(self, n_message_pos, n_message_neg, count_occurence_m_pos, count_occurence_m_neg):
        ### Valeur pratique ###

        #l'ensemble des mots uniques, sera utilisé pour le lissage de Laplace
        self.vocab = ( set(count_occurence_m_pos.keys()).union(set(count_occurence_m_neg.keys())) )

        #

        ###Valeur pour les proba pos ###

        # le nombre de message positif dans le corpus d'entraînement
        self.n_pos = n_message_pos
        #le prior positif
        self.prior_pos = n_message_pos / ( n_message_pos + n_message_neg )
        #le nombre d'occurence de chaque mots dans le corpus d'entraînement, sachant que le contexte (classe) est positif
        self.count_m_pos = count_occurence_m_pos
        #le nombre de mots dans les tweets positif
        self.nb_pos_word =  sum(count_occurence_m_pos.values())
        #la probabilité de chaque mot dans le corpus d'entraînement, sachant que le contexte (classe) est positif
        self.p_m_sachant_pos = {words : ( count_occurence_m_pos[words] / self.nb_pos_word ) for words in count_occurence_m_pos}

        ### Valeurs pour les probas neg ###

        # le nombre de message négatif dans le corpus d'entraînement
        self.n_neg = n_message_neg
        #le nombre d'occurence de chaque mots dans le corpus d'entraînement, sachant que le contexte (classe) est négatif
        self.count_m_neg = count_occurence_m_neg
        #le prior négatif
        self.prior_neg = self.n_neg / (self.n_pos + self.n_neg)
        #e nombre de mots dans les tweets négatif
        self.nb_neg_word =  sum(count_occurence_m_neg.values())
        #la probabilité de chaque mot dans le corpus d'entraînement, sachant que le contexte (classe) est négatif
        self.p_m_sachant_neg = {words : ( count_occurence_m_neg[words] / self.nb_neg_word ) for words in count_occurence_m_neg}
        

        pass


    pass

class Weight:
    def __init__(self,alpha_pos,alpha_neg):
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        pass


    pass

class Tableau :
    def __innit__(self):
        if ( os.path.exists("list_of_stops_words.txt") ):
            self.function_word_generated = read_simple_list_from_txt("stops_words")
        else:
            self.function_word_generated = []
        self.smiley = [
    # Visages heureux
    ":)", ":-)", ":D", ":-D", "8D", "8-D", "xD", "XD", "=D", "=)",  
    "^-^", "^^", "^_^", "n_n", "U_U", "UwU", "uwu", "OwO", "owo",  
    "(^_^)", "(o^_^o)", "(n_n)", "(^-^)", "(=^_^=)",  "<3", ":')" , ":'(" ,
   
    # Visages tristes et neutres
    ":(", ":-(", ":'(", ":'-(", "T_T", "TT_TT", "T-T", "-_-",  
    "v_v", "u_u", "._.", "-.-", "(._.)", "(>_<)", "(;-;)",  
   
    # Surprise et choc
    ":O", ":-O", ":o", ":-o", "O_O", "o_O", "O_o", "o_o", "(o_o)", "0_0",  
   
    # Clins d'oeil et taquineries
    ";)", ";-)", ";P", ";-P", ";D", ";-D",  
   
    # Expressions diverses
    ":P", ":-P", "XP", "X-P", "xP", "x-P", ":/", ":-/", ":|", ":-|",  
    "B)", "B-)",  "(?_?)", "(>_>)", "(<_<)"
]
        self.function_word_base = [
    # Manuellemenent rajouté
    "virginamerica","@virginamerica","my","jetblue","united","flight","usairways","americanair","southwestair","your","get","customer","service",

    "our","its","plane","time","im","airline","got","today","as","again",

    "gate","flights","back","fly","day","weather","airport","make","know","first",

    "home", "u", "ever", "work",

    "like", "please",

    "agent", "yes", "experience", "trip", "next",

    # Articles
    "a", "an", "the",
    
    # Conjonctions
    "and", "or", "for", "yet", "so", "because", "if", "while", "since", "unless", "but", "although",
    #retiré
    # "nor",
    
    # Prépositions
    "in", "on", "at", "by", "with", "about", "of", "for", "to", "from", "over", "under", "between", 
    "through", "into", "onto", "upon", "off", "along", "inside", "outside", "beneath",
    #retiré
    # "against", "out",

    # Pronoms personnels
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them","i","s",
    
    # Pronoms possessifs
    "mine", "yours", "his", "hers", "ours", "theirs",
    
    # Pronoms interrogatifs
    "who", "whom", "whose", "which", "what",
    
    # Adverbes fonctionnels
    "very", "just", "then", "there", "here", "well", "so", "now", "already", "still",
    #retiré
    #  "not",

    # Modaux (verbes auxiliaires)
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    
    # Déterminants
    "this", "that", "these", "those", "each", "some", "any", "no", "all", "both", "either",
    #retiré
    # "neither", "every", 

    # Particules
    "to", "up", "out", "off", "down",
    
    # Quantifieurs
    "few", "many", "much", "several", "most", 
    #retiré
    #"none",

    # Autres mots fonctionnels
    "does","been", "being", "has", "have", "had", "am", "is", "are", "was", "were", "be"
    #retiré
    #"do", "did", 
]
        pass
    pass





def main():

    print("start")
    train_df = read_file_from_csv("./tweets_train.csv")
    dev_df = read_file_from_csv("./tweets_dev.csv")
    test_df =read_file_from_csv("./tweets_test.csv")
    print("fin de lecture des fichiers")

    caution = Tableau()

    print("on génère le jeu d'entraînement")

    pos,neg = seperate_pos_nega(train_df)

    conca_pos = conca_str_in_list(pos)
    conca_neg = conca_str_in_list(neg)

    pos_words_list = [ tokenize(x) for x in ( conca_pos.split() ) ]
    neg_words_list = [ tokenize(x) for x in ( conca_neg.split() ) ]

    pos_final_sorted_dict = liste_occurrences(pos_words_list,caution)
    neg_final_sorted_dict = liste_occurrences(neg_words_list,caution)

    proba_train = Data()
    return


###################################################################
#################### Reader and writer of file ####################
###################################################################

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
        with open("list_of_" + file_name + ".txt", 'w',encoding="UTF-8") as f:
            for i in lst:
                f.write( i + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
    pass


def read_simple_list_from_txt(file_name):
    try:
        with open("list_of_" + file_name + ".txt", 'r') as f:
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


def send_dico_to_csv(dic,file_name):
    try:
        with open("dico_" + file_name + ".csv","w",newline="") as f:
            for i in dic:
                f.write(i + "," + str(dic[i]) + "\n")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    pass

def read_dico_from_csv(file_name):
    dico={}
    try:
        with open("dico_" + file_name + ".csv", 'r') as f:
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





#######################################################################
#################### Code for bayes classificateur ####################
#######################################################################


def conca_str_in_list(lst):
    conca_lst = " ".join(lst)
    return conca_lst


def liste_occurrences(l,caution: Tableau, type_of_caution_is_default = True):
    """
    prend une liste de mots l en entree et compte le nombre de fois que chaque mot apparait dans cette liste. 
    Elle retourne un dictionnaire res ou les cles sont les mots et les valeurs sont les occurrences de ces mots.
    type_of_caution_is_default indique 
    """
    res=dict()
    for i in l:
        if (type_of_caution_is_default and (i in caution.function_word_base) ) :
            continue
        if ( (not type_of_caution_is_default) and (i in caution.function_word_generated ) ):
            continue
        if not i in res:
            res[i]=1
        else:
            res[i]+=1
    return res


def sep_tweet_label(liste_tweet):
    label = []
    tweet = []
    for i in liste_tweet:
        label.append(i[0])
        tweet.append(i[1])
    return label,tweet


def seperate_pos_nega(liste_tweet):
    train_pos = []
    train_neg =[]
    for i in liste_tweet:
        if i[0] =="positive":
            train_pos.append(i[1])
        else:
            train_neg.append(i[1])
    return train_pos,train_neg


def lemmatize(word):
    if word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('ing') and len(word) > 4:
        return word[:-3]
    elif word.endswith('ed') and len(word) > 3:
        return word[:-2]
    elif word.endswith('s') and len(word) > 3:
        return word[:-1]
    return word


def tokenize(ori_words,words_to_remove,emoji):
    emoji_lower = emoji.lower()
    token = ori_words.lower()

    if ( (token in emoji_lower) or (token in emoji) ): #emoji, pas de traitements suplémentaires nécéssaires
        return token

    if ( token.isdigit() ):#que des numbres, on retire
        return 

    if ( token in words_to_remove ) :#est dans un mots pour lequel soit il ne carrie pas d'information (stop-words classique, ex: the,and..) soit on estime qu'il n'est pas important
        return 

    if ( token.isalpha() ) :#que des lettres, on lemmatize le mot
        return lemmatize(token)

    if ( token.startswith("https") ) :#retirer les liens hypertext, verifier qu'il commence par https
        return
    

    #si jamais il y avait un cara special colle au function word??????
    #essayer de laisser les ' pour voir ce que ça donne
    token_treated = re.sub(r'[^a-z]', '', token)
    if ( token_treated in words_to_remove ) :
        return 

    return token_treated


def compute_alpha_per_word(dataSet: Data, weight: Weight, base_alpha=1, rare_boost=3, freq_penalty=0.5):

    # Calcul des alpha pour la classe positive
    for i in dataSet.p_m_sachant_pos:

        if dataSet.p_m_sachant_pos[i] < 0.005:  # Mots très rares
            weight.alpha_pos[i] = base_alpha * rare_boost
            pass

        elif dataSet.p_m_sachant_pos > 0.05:  # Mots très fréquents
            weight.alpha_pos[i] = base_alpha * freq_penalty
            pass

        else: # Mots ni très fréquents, ni très rares
            weight.alpha_pos[i] = base_alpha
            pass
        pass

    # Calcul des alpha pour la classe négative
    for i in dataSet.p_m_sachant_neg:

        if dataSet.p_m_sachant_neg[i] < 0.005:  # Mots très rares
            weight.alpha_neg[i] = base_alpha * rare_boost
            pass

        elif dataSet.p_m_sachant_neg > 0.05:  # Mots très fréquents
            weight.alpha_neg[i] = base_alpha * freq_penalty
            pass

        else: # Mots ni très fréquents, ni très rares
            weight.alpha_neg[i] = base_alpha
            pass
        pass

    return 

def classification_tweet_basique(string,data_proba):# cette classification utilisa la formule de Bayes sans lissage

    p_prediction_pos = math.log(data_proba.prior_pos)  #var indiquant la proba que le tweet est positif sachant les mots du tweets
    p_prediction_neg = math.log(data_proba.prior_neg)  #var indiquant la proba que le tweet est negatif sachant les mots du tweets
    #traitement texte
    tweet = [ tokenize(x) for x in ( string.split() ) ]
    #calcul proba

    non_zero = 5.25e-6

    for i in tweet: #i est un mot du tweet
        if ( ( not ( i in data_proba.p_m ) ) ):#si il n'est pas dans le corpus general, il ne peut pas être dans les corpus positif ou corpus negatif
            p_prediction_pos += math.log(non_zero)
            p_prediction_neg += math.log(non_zero)
            continue

        
        if i == "":
            p_prediction_pos += math.log(non_zero)
            p_prediction_neg += math.log(non_zero)
            continue
        
        #calcul de P(POS|m)
        if (i in data_proba.p_m_sachant_pos):
            p_prediction_pos += math.log( data_proba.p_m_sachant_pos[ i ] )
            pass
        else:
            p_prediction_pos += math.log(non_zero)
            pass

        #calcul de P(NEG|m)
        if (i in data_proba.p_m_sachant_neg):
            p_prediction_neg += math.log( data_proba.p_m_sachant_neg[i] )
            pass
        else:
            p_prediction_neg += math.log(non_zero)
            pass

        if (False and ( ( p_prediction_neg > 1) or (p_prediction_pos > 1 ) ) ):
            print("\nERROR\n"*10)
            print(string)
            print(f"p_prediction_neg = {p_prediction_neg} et p_prediction_pos = {p_prediction_pos} et le mot analysé est {i}")
        pass
   
    if (p_prediction_pos > p_prediction_neg) :        #proba pos > prob neg
        return "positive"
    else:           #proba neg > proba pos
        return "negative"
    pass

def classification(string,data_proba: Data, weight: Weight):

    p_prediction_pos = math.log(data_proba.prior_pos)  #var indiquant la proba que le tweet est positif sachant les mots du tweets
    p_prediction_neg = math.log(data_proba.prior_neg)  #var indiquant la proba que le tweet est negatif sachant les mots du tweets
    #traitement texte
    tweet = [ tokenize(x) for x in ( string.split() ) ]

    for i in tweet:

        #On calcule P(m|POS)
        #on récupère l'alpha spécifique au mot
        alpha_pos = weight.alpha_pos.get(i, 1.0)
        #on calcule P(m|POS)
        word_prob_pos = ( ( data_proba.p_m_sachant_pos.get(i, 0.0) + alpha_pos ) / ( data_proba.nb_pos_word + alpha_pos * len(data_proba.vocab) ) )
        #On ajoute P(m|POS) à la prédiction
        p_prediction_pos += math.log( word_prob_pos )


        #On calcule P(m|NEG)
        #on récupère l'alpha spécifique au mot
        alpha_neg = weight.alpha_neg.get(i, 1.0)
        #on calcule P(m|NEG)
        word_prob_neg = ( ( data_proba.p_m_sachant_neg.get(i, 0.0) + alpha_neg ) / ( data_proba.nb_neg_word + alpha_neg * len(data_proba.vocab) ) )
        #On ajoute P(m|NEG) à la prédiction
        p_prediction_neg += math.log( word_prob_neg )

        pass


    if (p_prediction_pos > p_prediction_neg) :        #proba pos > prob neg
        return "positive"

    else:           #proba neg > proba pos
        return "negative"

    return







#############################################################
################# Appel de la fonction main #################
#############################################################

main()
