from ast import Dict, Pass
import re
import pandas as pd
import math
import os
from typing import Dict
from time import sleep
import random
import sys
import time
import queue
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import multiprocessing
from multiprocessing import Queue, Manager, Lock
import csv

class Data:
    """
    Classe contenant toutes les donnéees du jeu d'entrainement (après traitement)

    Attributs :
    - n_pos : int
        Le nombre de messages positifs dans le corpus d'entraînement.
    - n_neg : int
        Le nombre de messages négatifs dans le corpus d'entraînement.
    - prior_pos : float
        La probabilité a priori d'un message positif dans le corpus.
    - prior_neg : float
        La probabilité a priori d'un message négatif dans le corpus.
    - count_m_pos : Dict[str, int]
        Dictionnaire représentant le nombre d'occurrences de chaque mot dans les messages positifs.
    - count_m_neg : Dict[str, int]
        Dictionnaire représentant le nombre d'occurrences de chaque mot dans les messages négatifs.
    - nb_pos_word : int
        Le nombre total de mots dans les messages positifs.
    - nb_neg_word : int
        Le nombre total de mots dans les messages négatifs.
    - p_m_sachant_pos : Dict[str, float]
        Dictionnaire représentant la probabilité de chaque mot étant donné que le message est positif.
    - p_m_sachant_neg : Dict[str, float]
        Dictionnaire représentant la probabilité de chaque mot étant donné que le message est négatif.
    - vocab : set
        L'ensemble des mots uniques présents dans l'ensemble du corpus (positif et négatif), utilisé pour le lissage de Laplace.

    Méthodes :
    - __init__(self, n_message_pos: int, n_message_neg: int, 
               count_occurence_m_pos: Dict[str, int], count_occurence_m_neg: Dict[str, int])
        Initialise les valeurs.
    """

    def __init__(self, 
                 n_message_pos: int, 
                 n_message_neg: int,
                 count_occurence_m_pos: Dict[str, int], 
                 count_occurence_m_neg: Dict[str, int]):
        """
        Initialise un objet Data qui contient les informations nécessaires pour le calcul du modèle.

        Args:
            n_message_pos (int) : Nombre de messages positifs dans le corpus d'entraînement.
            n_message_neg (int) : Nombre de messages négatifs dans le corpus d'entraînement.
            count_occurence_m_pos (Dict[str, int]) : Dictionnaire avec les mots comme clés et les occurrences de chaque mot 
                                                     dans les messages positifs comme valeurs.
            count_occurence_m_neg (Dict[str, int]) : Dictionnaire avec les mots comme clés et les occurrences de chaque mot 
                                                     dans les messages négatifs comme valeurs.
        """

        ### Valeur pratique ###
        # L'ensemble des mots uniques dans le corpus (positif et négatif)
        self.vocab = set(count_occurence_m_pos.keys()).union(set(count_occurence_m_neg.keys()))

        ### Valeur pour les proba pos ###

        # Le nombre de messages positifs
        self.n_pos: int = n_message_pos
        # Le prior positif : probabilité d'un message positif
        self.prior_pos: float = n_message_pos / ( n_message_pos + n_message_neg )
        # Dictionnaire des occurrences des mots dans les messages positifs
        self.count_m_pos: Dict[str, int] = count_occurence_m_pos
        # Nombre total de mots dans les messages positifs
        self.nb_pos_word: int = sum( count_occurence_m_pos.values() )
        # Dictionnaire des probabilités de chaque mot étant donné que le message est positif
        self.p_m_sachant_pos: Dict[str, float] = {
            words: ( count_occurence_m_pos[words] / self.nb_pos_word ) for words in count_occurence_m_pos
        }

        ### Valeurs pour les probas neg ###

        # Le nombre de messages négatifs
        self.n_neg: int = n_message_neg
        # Le prior négatif : probabilité d'un message négatif
        self.prior_neg: float = n_message_neg / ( n_message_pos + n_message_neg )
        # Dictionnaire des occurrences des mots dans les messages négatifs
        self.count_m_neg: Dict[str, int] = count_occurence_m_neg
        # Nombre total de mots dans les messages négatifs
        self.nb_neg_word: int = sum( count_occurence_m_neg.values() )
        # Dictionnaire des probabilités de chaque mot étant donné que le message est négatif
        self.p_m_sachant_neg: Dict[str, float] = {
            words: ( count_occurence_m_neg[words] / self.nb_neg_word ) for words in count_occurence_m_neg
        }
        pass


    pass

class Weight:
    def __init__(self,alpha_pos,alpha_neg):
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        pass


    pass

stop_words = [# Articles
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


class Tableau:
    """
    Classe `Tableau` utilisée pour stocker des listes de mots fonctionnels et de smileys,
    afin de faciliter le prétraitement des messages.

    Attributs :
    - function_word_generated : List[str]
        Liste de mots fonctionnels (mots vides) chargés depuis un fichier texte si celui-ci existe.
    - smiley : List[str]
        Liste prédéfinie de chaînes représentant des émojis et émoticônes textuels pour la détection d'émotions.
    - function_word_base : List[str]
        Liste étendue et manuellement enrichie de mots fonctionnels (articles, conjonctions, prépositions, pronoms, modaux, etc.)
    """

    def __init__(self):
        """
        Initialise un objet `Tableau`, en chargeant une liste de mots vides depuis un fichier
        si celui-ci existe, et en définissant des listes de smileys et de mots fonctionnels.
        """
        if ( os.path.exists("list_of_stops_words.txt") ):
            self.function_word_generated = read_simple_list_from_txt("stops_words")
        else:
            self.function_word_generated = []
            pass
        #liste de smiley
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
        self.smiley_lower = [x.lower() for x in self.smiley]
        #liste de stop words fait manuellements
        self.function_word_base = [
    # Manuellemenent rajouté
    "virginamerica","@virginamerica","my","jetblue","united","flight","usairways","americanair","southwestair","your","get","customer","service",

    "our","its","plane","time","im","airline","got","today","as","again",

    "gate","flights","back","fly","day","weather","airport","make","know","first",

    "home", "u", "ever", "work",

    "like", "please",

    "agent", "yes", "experience", "trip", "next",

    "thing","airways","second","delivered",

    "pretty","n","flightd","matter","folks","planned","drinks",

    "coach","trust","representative","knowing","mad","apologies","leaves","showed",

    "based","understanding","destinations","carriers","bummer","alliance","operations","cle","hit","fill",

    "pacific","pocket","informed","pit","ca","die","trained","payment","cancun","stress","juan","deicing","feature","large",

    "go","missed","another",
    #retiré
    #"looking",

    "checked","tried",
    #retiré
    #"talk","called",

    "too","sent","airlines","boarding","baggage","ua","attendant","planes","pilots","company","charlotte","american","flyfi","speak","sit","more","even",
    #"flighted","southwest","flightr",

    "delay","travel","class","half","different",

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






# ----------------------------
# Formatage du temps (identique)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# ----------------------------
# Fonction pour mettre à jour le meilleur résultat
def update_best_seed_func(seed, precision, window):
    window.update_best_seed(seed, precision)



# ----------------------------
# Classe GUI 
class ProgressWindow(QWidget):
    def __init__(self, max_seed, tested_seeds_counter, update_queue, timing_info, max_result):
        super().__init__()
        self.max_seed = max_seed
        self.tested_seeds_counter = tested_seeds_counter
        self.update_queue = update_queue
        self.timing_info = timing_info
        self.max_result = max_result
        self.best_seed = None
        self.best_precision = None

        self.setWindowTitle("Progression des calculs")
        self.setGeometry(100, 100, 400, 150)
        layout = QVBoxLayout()

        self.label = QLabel("Seeds testés: 0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        self.time_per_seed_label = QLabel("Temps moyen par seed: 0.00s")
        self.eta_label = QLabel("Estimation du temps restant: ...")
        self.best_seed_label = QLabel("Meilleure Seed: N/A | Précision: N/A")

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.time_per_seed_label)
        layout.addWidget(self.eta_label)
        layout.addWidget(self.best_seed_label)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)

    def update_best_seed(self, seed, precision):
        self.best_seed = seed
        self.best_precision = precision
        self.best_seed_label.setText(f"Meilleure Seed: {self.best_seed} | Précision: {self.best_precision:.4f}")

    def update_gui(self):
        while not self.update_queue.empty():
            tested = self.update_queue.get()
            self.label.setText(f"Seeds testés: {tested} / {self.max_seed}")
            self.progress_bar.setValue(int((tested / self.max_seed) * 100))

            if self.timing_info["count"] > 0:
                avg_time = self.timing_info["total_time"] / self.timing_info["count"]
                remaining = self.max_seed - tested
                eta = remaining * avg_time
                formatted_eta = format_time(eta)

                self.time_per_seed_label.setText(f"Temps moyen par seed: {avg_time:.2f}s")
                self.eta_label.setText(f"Estimation du temps restant: {formatted_eta}")

            # Met à jour le meilleur résultat affiché
            self.update_best_seed(self.max_result["seed"], self.max_result["value"])
            pass
        pass
    pass






# ----------------------------
# Fonction principale
def main():

    #on charge les jeu de données
    print("start")
    train_df = read_file_from_csv("./tweets_train.csv")
    dev_df = read_file_from_csv("./tweets_dev.csv")
    test_df = read_file_from_csv("./tweets_test.csv")
    print("fin de lecture des fichiers")

    caution = Tableau()


    print("on génère le jeu d'entraînement")

    proba_train = generate_training_data(train_df, caution)
    save_word_stats_to_csv(proba_train, filename= "info_Data_train.csv")#utilisé pour déterminer quels sont les mots à retirer (stop - words) pour function_word_base

    print("le jeu d'entraînement a fini d'être généré")
    
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    ######### Code pour optimiser les prédictions au possibles, utilises multiprocessing #########
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################

    #test le rebalançage du jeu de donnée d'entraînement pour voir si on a de meilleur résultat
    if False: #testé les 400000 premières seed prends environ 12h
        min_seed = 0
        max_seed = 1000
        num_processes = 14
        seed = list(
            range( min_seed, max_seed + num_processes - ( (max_seed-min_seed) % num_processes) ) # pour s'assurer que toutes les seeds sont testé
            )
        
        seeds_per_process = len(seed) // num_processes # si len(seed) % num_processes était différnt de 0, alors certaine seeds ne serait pas traité
        type_of_balancing = "oversample"

        manager = Manager()
        max_result = manager.dict({"seed": 0, "value": -1.5})
        tested_seeds_counter = manager.list([0])
        update_queue = Queue()
        lock = Lock()
        timing_info = manager.dict({"total_time": 0.0, "count": 0})

        #interface graphique pour visualiser le progrès

        app = QApplication(sys.argv)

        window = ProgressWindow(
            len(seed), tested_seeds_counter, update_queue, timing_info, max_result
        )
        window.show()

        p = multiprocessing.Process(target=start_processes, args=(
            seed, max_result, lock, train_df, dev_df, caution,
            tested_seeds_counter, update_queue, timing_info,
            num_processes, seeds_per_process),
            kwargs={"method_of_balancing" : type_of_balancing})
        p.start()
        sys.exit(app.exec_())
        pass
    

    #on essaie de trouver les meilleurs non-zero, voir README pour plus de détail sur la méthode qui a été suivie
    if False:
        non_zero_list = [(k * 1e-9) for k in range(1, 100000)]
        num_processes = 5
        start_processes2(non_zero_list, dev_df, proba_train, caution, num_processes)
        pass

    #on essaie de trouver les meilleurs stops-words
    if ( not ( os.path.exists("list_of_stops_words.txt") ) ):#inutile de rechercher les meilleurs stops words si ils ont déjà été calculé

        #on s'assure que proba_train.vocab est divisble en 13 morceaux
        proba_train.vocab.remove("")
        num_process = 13

        vocab_list = list(proba_train.vocab)
        vocab_chunks = split_list(vocab_list, num_process)
        manager = multiprocessing.Manager()
        shared_final_words = manager.list()
        lock = manager.Lock()
        processes = []

        for chunk in vocab_chunks:
            p = multiprocessing.Process(
                target=find_best_stop_word,
                args=(chunk, train_df, dev_df, shared_final_words, lock)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        send_simple_list_to_txt(list(shared_final_words), "stops_words")#on sauvegarde les stops-words dans un fichier texte
        pass


    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    #########Fin  du code pour optimiser les prédictions au possibles, utilises multiprocessing#########
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    
    print("\n\n\n\n\n")
    print("éssais sur le dev set")
    a = naive_bayesian_prediction(dev_df, proba_train, caution)

    print("\n\n\n\n\n")
    print("éssais sur le test set")
    
    a = naive_bayesian_prediction(test_df, proba_train, caution)
    sleep(3)


    ar=10
    #a = naive_bayesian_prediction(test_df, proba_train_recalibrated, caution, mode=3)
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






#######################################################################
#################### Code for bayes classificateur ####################
#######################################################################


def seperate_pos_nega(liste_tweet: list[list[str, str]]) -> tuple[list[str], list[str]]:
    """
    Sépare une liste de tweets en deux listes distinctes 

    Chaque élément de `liste_tweet` est une liste ou un tuple contenant deux éléments :
    - i[0] : un label de polarité ("positive" ou "negative")
    - i[1] : le texte du tweet

    Si le label est "positive", le texte est ajouté à la liste des positifs.
    Sinon, il est ajouté à la liste des négatifs.

    Args:
        liste_tweet (list[list[str, str]]): Liste de tweets annotés.

    Returns:
        tuple[list[str], list[str]]: Deux listes de tweets :
                                     - La première contenant les tweets positifs.
                                     - La seconde contenant les tweets négatifs.
    """
    train_pos = []  # Liste pour stocker les tweets positifs
    train_neg = []  # Liste pour stocker les tweets négatifs

    for i in liste_tweet:
        if i[0] == "positive":     # Si le label est "positive"
            train_pos.append(i[1])  # Ajouter le texte à la liste des positifs
        else:
            train_neg.append(i[1])  # Sinon, l'ajouter à la liste des négatifs

    return train_pos, train_neg  # Retourner les deux listes


def conca_str_in_list(lst: list[str]) -> str:
    """
    Concatène les chaînes de caractères d'une liste en une seule chaîne, séparée par des espaces.

    Args:
        lst (list[str]): Liste de chaînes de caractères à concaténer.

    Returns:
        str: Une seule chaîne contenant tous les éléments de la liste, séparés par des espaces.
    """
    # Utilise str.join pour concaténer les éléments avec un espace
    return " ".join(lst)


def liste_occurrences(words_list: list[str], caution: Tableau, use_default_caution=True) -> dict:
    """
    Prend une liste de mots `words_list` en entrée et compte le nombre de fois que chaque mot
    apparaît dans cette liste. Elle retourne un dictionnaire `res` où :
      - les clés sont les mots,
      - les valeurs sont leurs occurrences.

    Le paramètre `caution` est une instance de la classe `Tableau` contenant
    des listes de mots à ignorer selon le contexte (use_default_caution).

    Si `use_default_caution` est True :
        - on ignore les mots présents dans `tableau.function_word_base`.

    Si `use_default_caution` est False :
        - on ignore les mots présents dans `tableau.function_word_generated`.

    Args:
        words_list (list[str]): Liste de mots à analyser.
        tableau (Tableau): Objet contenant des listes de mots à ignorer.
        use_default_caution (bool): Détermine quel type de mots ignorer (base ou généré).

    Returns:
        dict: Dictionnaire des occurrences de mots (hors mots ignorés).
    """
    word_counts = dict()  # Dictionnaire pour stocker les occurrences de mots

    for word in words_list:

        # Si le type de filtre est "default" et que le mot est dans les mots à ignorer
        if use_default_caution and (word in caution.function_word_base):
            continue  # Ignorer ce mot

        # Si le type de filtre est "generated" et que le mot est dans les mots à ignorer
        if not use_default_caution and (word in caution.function_word_generated):
            continue  # Ignorer ce mot

        # Incrémenter le compteur du mot dans le dictionnaire
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

    return word_counts  # Retourner le dictionnaire des occurrences


def sep_tweet_label(tweets_with_labels: list[list[str]]) -> tuple[list[str], list[str]]:
    """
    Sépare une liste de tweets en deux listes distinctes : 
    une contenant les labels (par exemple : "positive", "negative") et l'autre les textes des tweets.

    Args:
        tweets_with_labels (list[list[str]]): Liste de tweets, chaque élément étant une sous-liste contenant :
                                               - le label du tweet (ex: "positive", "negative"),
                                               - le texte du tweet.

    Returns:
        tuple[list[str], list[str]]: Deux listes :
                                     - La première contenant les labels des tweets.
                                     - La seconde contenant les textes des tweets.
    """
    labels = []  # Liste pour stocker les labels des tweets
    tweets = []  # Liste pour stocker les textes des tweets

    for tweet in tweets_with_labels:
        labels.append(tweet[0])  # Ajouter le label du tweet à la liste des labels
        tweets.append(tweet[1])  # Ajouter le texte du tweet à la liste des tweets

    return labels, tweets  # Retourner les deux listes



def lemmatize(word: str) -> str: #à retravailler, impact sur la précision moyen, pas grand, tends vers perte de précision
    #disabling the fonction because it lower the accuracy
    if True:
        return word

    # Exceptions courantes
    exceptions = {
        "better": "good",
        "worse": "bad",
        "children": "child",
        "mice": "mouse",
        "feet": "foot",
        "geese": "goose",
        "men": "man",
        "women": "woman"
    }
    if word in exceptions:
        return exceptions[word]

    # "ies" -> "y"  (e.g., studies → study)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # "ing" → base (e.g., running → run)
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        # handle doubling: "running" → "run"
        if base[-1] == base[-2]:
            base = base[:-1]
        return base

    # "ed" → base (e.g., jumped → jump)
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        # handle doubled consonant: "stopped" → "stop"
        if base[-1] == base[-2]:
            base = base[:-1]
        return base

    # Remove plural 's', but avoid over-truncating
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def tokenize(original_word: str, tableau: Tableau, use_default_caution: bool = True) -> str | None:
    """
    Nettoie et transforme un mot selon plusieurs règles de filtrage linguistique :

    - Convertit le mot en minuscule
    - Ignore certains types de tokens (emojis, chiffres, liens, mots fonctionnels)
    - Applique une lemmatisation sur les mots valides
    - Supprime les caractères non alphabétiques si nécessaire

    Args:
        original_word (str): Le mot original à traiter.
        tableau (Tableau): Instance contenant les listes de mots à ignorer.
        use_default_caution (bool): Détermine quel type de mots fonctionnels utiliser pour le filtrage.

    Returns:
        str | None: Le mot lemmatisé, ou None s’il est filtré/ignoré.
    """
    token = original_word.lower()  # Mettre le mot en minuscule

    # Emojis / smileys simples : on les conserve tels quels
    if token in tableau.smiley_lower or token in tableau.smiley:
        return token

    # Token numérique : à ignorer
    if token.isdigit():
        return None

    # Lien hypertexte (ex : https://...) : à ignorer
    if token.startswith("https"):
        return None

    # Stop-words à ignorer selon le type de caution
    if use_default_caution and token in tableau.function_word_base:
        return None
    if not use_default_caution and token in tableau.function_word_generated:
        return None

    # Mot uniquement alphabétique et qui ne sont pas dans les stops-words : on lemmatise
    if token.isalpha():
        return lemmatize(token)

    # Autres cas : nettoyage du mot (suppression des caractères non alphabétiques)
    token_treated = re.sub(r"[^a-z]", '', token)

    # Nouveau filtrage après nettoyage
    if use_default_caution and token in tableau.function_word_base:
        return None
    if not use_default_caution and token in tableau.function_word_generated:
        return None

    return lemmatize(token_treated)  # Lemmatise et retourne le mot nettoyé


def generate_training_data(string: list[str], caution: Tableau, caution_type=True):
    """
    Génère un objet Data contenant les statistiques de mots issus des tweets positifs et négatifs.

    Cette fonction :
    - Sépare les tweets en positifs et négatifs.
    - Concatène chaque groupe de tweets en une seule chaîne.
    - Tokenize les mots de chaque groupe en appliquant un nettoyage et une lemmatisation.
    - Compte les occurrences des mots en excluant ceux à ignorer via l'objet `caution`.
    - Retourne un objet `Data` contenant ces informations.

    Args:
        string (list[str]): Liste de tweets avec leurs labels sous forme [["positive", "texte1"], ["negative", "texte2"], ...].
        caution (Tableau): Objet contenant des listes de mots à exclure du traitement.
        caution_type (bool): Si True, utilise `function_word_base` ; sinon, `function_word_generated`.

    Returns:
        Data: Objet contenant le nombre de tweets positifs/négatifs et les dictionnaires d’occurrences.
    """

    # Séparation des tweets en positifs et négatifs
    pos, neg = seperate_pos_nega(string)

    # Concaténation des tweets positifs et négatifs en une seule chaîne de texte
    conca_pos = conca_str_in_list(pos)
    conca_neg = conca_str_in_list(neg)

    # Découpage en mots + traitement via la fonction tokenize (nettoyage, filtrage, lemmatisation)
    pos_words_list = [ tokenize( word, caution, use_default_caution=caution_type ) for word in conca_pos.split() ]
    neg_words_list = [ tokenize( word, caution, use_default_caution=caution_type ) for word in conca_neg.split() ]

    # Calcul des occurrences de chaque mot dans les deux listes
    pos_final_sorted_dict = liste_occurrences(pos_words_list, caution, use_default_caution = caution_type)
    neg_final_sorted_dict = liste_occurrences(neg_words_list, caution, use_default_caution = caution_type)

    # Retour d'un objet Data à partir des informations fournies qui ont été traité
    return Data(len(pos), len(neg), pos_final_sorted_dict, neg_final_sorted_dict)

def naive_bayesian_prediction(data_set_to_analyse: list[list[str]], data_proba: Data, caution: Tableau, mode=1,verbose = True) -> float:
    """
    Effectue une prédiction bayésienne sur un ensemble de tweets étiquetés, 
    en utilisant différentes variantes du modèle naïf bayésien.
    mode = 3 et mode = 4 ne fonctionne pas
    Args:
        data_set_to_analyse (list[list[str]]): Liste de tweets avec leurs labels, sous la forme [[label, texte], ...].
        data_proba (Data): Objet contenant les probabilités ou les occurrences pour chaque classe (positif/négatif).
        caution (Tableau): Objet contenant les mots à ignorer (stop-words ou autres filtres).
        mode (int): Définit le type de modèle utilisé :
            - 1 : utilise `classification_tweet_basique` avec `function_word_base`
            - 2 : utilise `classification_tweet_basique` avec `function_word_generated`
            - 3 : utilise `classification` avec `function_word_base`
            - 4 : utilise `classification` avec `function_word_generated`
        verbose (Bool) ; affiche la matrice d'occurence si verbose = True

    Returns:
        float: Accuracy du modèle sur le jeu de données fourni.
    """

    # Séparation des labels et des tweets
    label, tweet = sep_tweet_label(data_set_to_analyse)

    match mode:
        case 1:
            # Classification simple avec les stop-words de base
            prediction = [classification_tweet_basique(x, data_proba, caution) for x in tweet]
            accuracy = matrice_confusion(label, prediction,verbose = verbose)

        case 2:
            # Classification simple avec les stop-words générés automatiquement
            prediction = [classification_tweet_basique(x, data_proba, caution, caution_type=False) for x in tweet]
            accuracy = matrice_confusion(label, prediction,verbose = verbose)
            """
        case 3:
            # Classification avancée avec les stop-words de base
            prediction = [classification(x, data_proba, caution) for x in tweet]
            accuracy = matrice_confusion(label, prediction,verbose = verbose)

        case 4:
            # Classification avancée avec les stop-words générés automatiquement
            prediction = [classification(x, data_proba, caution, caution_type=False) for x in tweet]
            accuracy = matrice_confusion(label, prediction,verbose = verbose)
            """
    return accuracy

def compute_alpha_per_word(dataSet: Data, weight: Weight, base_alpha=1, rare_boost=4, freq_penalty=0.3):

    # Calcul des alpha pour la classe positive
    for i in dataSet.p_m_sachant_pos:

        if ( i == None ):
            continue

        if dataSet.p_m_sachant_pos[i] < 0.005:  # Mots très rares
            weight.alpha_pos[i] = base_alpha * rare_boost
            pass

        elif dataSet.p_m_sachant_pos[i] > 0.05:  # Mots très fréquents
            weight.alpha_pos[i] = base_alpha * freq_penalty
            pass

        else: # Mots ni très fréquents, ni très rares
            weight.alpha_pos[i] = base_alpha
            pass
        pass

    # Calcul des alpha pour la classe négative
    for i in dataSet.p_m_sachant_neg:

        if ( i == None ):
            continue

        if dataSet.p_m_sachant_neg[i] < 0.005:  # Mots très rares
            weight.alpha_neg[i] = base_alpha * rare_boost 
            pass

        elif dataSet.p_m_sachant_neg[i] > 0.05:  # Mots très fréquents
            weight.alpha_neg[i] = base_alpha * freq_penalty 
            pass

        else: # Mots ni très fréquents, ni très rares
            weight.alpha_neg[i] = base_alpha 
            pass
        pass

    return 




def classification_tweet_basique(string: str, data_proba: Data, caution: Tableau, caution_type=True, non_zero_para = 9.63e-6 ) -> str:
    """
    Classifie un tweet en "positive" ou "negative" en utilisant un modèle Naïf Bayésien.

    Args:
        string (str): Le tweet à analyser.
        data_proba (Data): Objet contenant les probabilités a priori et les probabilités conditionnelles des mots.
        caution (Tableau): Objet contenant les listes de mots à exclure (stop-words, emojis, etc.).
        caution_type (bool): Si True, utilise `function_word_base`, sinon `function_word_generated`.

    Returns:
        str: "positive" si la probabilité estimée que le tweet soit positif est plus grande, sinon "negative".
    """

    # Log des probabilités a priori (P(positive), P(negative))
    p_prediction_pos = math.log(data_proba.prior_pos)
    p_prediction_neg = math.log(data_proba.prior_neg)

    # Tokenisation du tweet : nettoyage, filtrage, lemmatisation
    tweet = [ tokenize( word, caution, use_default_caution = caution_type ) for word in string.split()]

    # Petites valeurs pour éviter log(0) ou log(1) pour tous les mots dont on a pas P(m|contexte)
    non_zero =non_zero_para  #non_zero_para        
    non_zero_empty_word_pos = 9.55e-6
    non_zero_empty_word_neg = 1.022e-5
    non_zero_absent_vocab_pos = 9.63e-6
    non_zero_absent_vocab_neg = 9.59e-6
    non_zero_pos_absent =  9.62e-6    # mots absent dans 1 dictionnaires
    non_zero_neg_absent = 6.999e-5

    for word in tweet:

        # Mot vide (résultat possible de nettoyage)
        if word == "" or word == None:
            p_prediction_pos += math.log(non_zero_empty_word_pos)
            p_prediction_neg += math.log(non_zero_empty_word_neg)
            continue

        # Mot absent du vocabulaire général
        if word not in data_proba.vocab:
            p_prediction_pos += math.log(non_zero_absent_vocab_pos)
            p_prediction_neg += math.log(non_zero_absent_vocab_neg)
            continue

        # Ajout de la log probabilité conditionnelle pour POS
        if word in data_proba.p_m_sachant_pos:
            p_prediction_pos += math.log(data_proba.p_m_sachant_pos[word])
        else:
            p_prediction_pos += math.log(non_zero_pos_absent)

        # Ajout de la log probabilité conditionnelle pour NEG
        if word in data_proba.p_m_sachant_neg:
            p_prediction_neg += math.log(data_proba.p_m_sachant_neg[word])
        else:
            p_prediction_neg += math.log(non_zero_neg_absent)

        # Debugging optionnel, normalement, cette condition ne devrait jamais être vraie
        if False and ((p_prediction_neg > 0) or (p_prediction_pos > 0)):
            print("\nERROR\n" * 10)
            print(string)
            print(f"p_prediction_neg = {p_prediction_neg} et p_prediction_pos = {p_prediction_pos} et le mot analysé est {word}")
   
    if (p_prediction_pos > p_prediction_neg) :        #proba pos > prob neg
        return "positive"
    else:           #proba neg > proba pos
        return "negative"
    pass



def classification(string: str, data_proba: Data, caution: Tableau, caution_type = True) -> str:
    """
    Classifie un tweet en "positive" ou "negative" en utilisant un modèle Naïf Bayésien 
    avec lissage adaptatif (alpha spécifique par mot).

    Args:
        string (str): Le tweet à analyser.
        data_proba (Data): Objet contenant les probabilités a priori et conditionnelles.
        caution (Tableau): Objet contenant des listes de mots à exclure.
        caution_type (bool): Si True, utilise function_word_base, sinon function_word_generated.

    Returns:
        str: "positive" ou "negative" selon la classe prédite.
    """

    # Calcul des coefficients de lissage alpha pour chaque mot (selon data_proba)
    weight = Weight({}, {})
    compute_alpha_per_word(data_proba, weight)

    # Initialisation avec les probabilités a priori (sans log)
    p_prediction_pos = data_proba.prior_pos
    p_prediction_neg = data_proba.prior_neg

    # Tokenisation du tweet
    tweet_tokens = [ tokenize( word, caution, use_default_caution = caution_type ) for word in string.split()]

    for word in tweet_tokens:
        # Lissage alpha spécifique au mot (défaut = 1.0 si absent)
        alpha_pos = weight.alpha_pos.get(word, 1.0)
        alpha_neg = weight.alpha_neg.get(word, 1.0)

        # Probabilité conditionnelle P(word | POS) avec lissage
        prob_word_given_pos = (
            (data_proba.p_m_sachant_pos.get(word, 0.0) + alpha_pos)
            / (data_proba.nb_pos_word + alpha_pos * len(data_proba.vocab))
        )
        p_prediction_pos *= prob_word_given_pos

        # Probabilité conditionnelle P(word | NEG) avec lissage
        prob_word_given_neg = (
            (data_proba.p_m_sachant_neg.get(word, 0.0) + alpha_neg)
            / (data_proba.nb_neg_word + alpha_neg * len(data_proba.vocab))
        )
        p_prediction_neg *= prob_word_given_neg

        # Debug facultatif
        if False:
            print(f"Mot analysé : {word}")
            print(f"alpha_pos = {alpha_pos}, alpha_neg = {alpha_neg}")
            print(f"P(m|POS) = {prob_word_given_pos}, P(m|NEG) = {prob_word_given_neg}")
            print(f"Prédiction intermédiaire POS = {p_prediction_pos}")
            print(f"Prédiction intermédiaire NEG = {p_prediction_neg}")
            print("=> Classe prédite:", "positive" if p_prediction_pos > p_prediction_neg else "negative")
            print("-" * 40)

    # Prédiction finale
    return "positive" if p_prediction_pos > p_prediction_neg else "negative"



def matrice_confusion(lst_correct_label: list[str], lst_prediction: list[str], verbose: bool = True):
    """
    Calcule et affiche la matrice de confusion ainsi que les précisions des prédictions.

    Paramètres :
    - lst_correct_label : liste des vraies étiquettes ('positive' ou 'negative')
    - lst_prediction : liste des étiquettes prédites correspondantes
    - verbose : si False, n'affiche pas les résultats ni la matrice de confusion, utilisé pendant les test pour trouver les valeurs optimales

    Retourne :
    - accuracy (float) : pourcentage de bonnes prédictions
    """

    # Initialisation des compteurs
    true_positif = false_positif = false_negative = true_negative = 0  

    # calcul de la matrice d'occurence
    for i in range(len(lst_prediction)):
        if lst_prediction[i] == lst_correct_label[i]:
            if lst_prediction[i] == "positive": # lst_correct_label[i] == "positive" and lst_prediction[i] == "positive"
                true_positif += 1  # Vrai Positif
            else: # lst_correct_label[i] == "negative" and lst_prediction[i] == "negative"
                true_negative += 1  # Vrai Négatif
        else:
            if lst_prediction[i] == "positive": # lst_correct_label[i] == "negative" and lst_prediction[i] == "positive"
                false_positif += 1  # Faux Positif
            else: # lst_correct_label[i] == "positive" and lst_prediction[i] == "negative"
                false_negative += 1  # Faux Négatif

    #on évite les divisions par zéro, et oui il y a eu des cas avec division par zéro lors des tests
    if ( ( true_positif + false_positif ) == 0 ) or ( ( true_negative + false_negative ) == 0 ):
        if verbose:
            print("               Prédit Negative   Prédit Positive")
            print(f"Réel Negative    {true_negative:<10}       {false_positif:<10}")
            print(f"Réel Positive    {false_negative:<10}       {true_positif:<10}")
        return 0.0  

    # Calcul de la précision pour les prédictions positives
    precision_pos = true_positif / (false_positif + true_positif)

    # Calcul de la précision pour les prédictions négatives
    precision_neg = true_negative / (true_negative + false_negative)

    #calcul de la précision du modèle pour touts les tweets
    accuracy = (100 * (true_positif + true_negative ) / len(lst_correct_label))

    #affichage des résultats
    if verbose:
        print(f"Précision des prédictions positives : {100*precision_pos:.5f}")
        print(f"Précision des prédictions négatives : {100*precision_neg:.5f}")
        print(f"L'accuracy est de : { accuracy: .5f}")
        # Affichage de la matrice sous forme de tableau
        print("Matrice de Confusion :")
        print("               Prédit Negative   Prédit Positive")
        print(f"Réel Negative    {true_negative:<10}       {false_positif:<10}")
        print(f"Réel Positive    {false_negative:<10}       {true_positif:<10}")
        print(f"il y a {false_positif + false_negative} fausse prédiction")

    return accuracy

def generate_training_data_with_calibrage(string: str, caution: Tableau, random_seed = 42, balancing_method = "oversample", caution_type = True):
    """
    Génére les données d'entraînement en rééquilibrant d'abord les tweets selon la méthode choisie
    puis en traitant les textes (tokenisation, suppression des mots peu informatifs, etc.)
    et en comptant les occurrences de chaque mot.

    Paramètres :
    - string (str) : Liste de tweets avec leurs labels au format [[label, tweet], ...].
    - caution (Tableau) : Objet contenant les listes de mots à ignorer selon le type de traitement.
    - random_seed (int) : Seed aléatoire pour le rééquilibrage des classes (par défaut 42).
    - balancing_method (str) : Méthode de rééquilibrage, par exemple "oversample" ou "undersample".
    - caution_type (bool) : Si True, utilise les mots dans `function_word_base` comme mots à ignorer, sinon utilise `function_word_generated`.

    Retour :
    - Data : Un objet contenant les statistiques d'entraînement (nombre de tweets, dictionnaires d’occurrences, etc.).
    """

    # Rééquilibrage du dataset (par exemple via sur-échantillonnage)
    pos, neg = rebalance_dataset(string, seed=random_seed, method=balancing_method)

    # Concaténation des tweets positifs et négatifs
    conca_pos = conca_str_in_list(pos)
    conca_neg = conca_str_in_list(neg)

    # Tokenisation des tweets (suppression des mots inutiles, liens, chiffres, etc.)
    pos_words_list = [tokenize(x, caution, use_default_caution=caution_type) for x in conca_pos.split()]
    neg_words_list = [tokenize(x, caution, use_default_caution=caution_type) for x in conca_neg.split()]

    # Comptage des occurrences de mots dans les tweets positifs et négatifs
    pos_words_occurence = liste_occurrences(pos_words_list, caution)
    neg_words_occurence = liste_occurrences(neg_words_list, caution)

    # Création de l'objet contenant les données préparées pour le modèle bayésien
    return Data(len(pos), len(neg), pos_words_occurence, neg_words_occurence)

#
#
##
#
#"
#
#
#
#
#
#
#
#
#Rédiger la doc pour cette fonction
#
#
#
#
#
#
#
#
def rebalance_dataset(data: list[str, str], method='oversample', seed=40):
    """
    data (list of tuples): [(label, text), ...]
    """
    random.seed(seed)
    train_pos = []
    train_neg =[]
    for i in data:
        if i[0] =="positive":
            train_pos.append(i[1])
        else:
            train_neg.append(i[1])
            pass
        continue
    sample_holder = {"positive" : train_pos, "negative" : train_neg}
    sample_size = {"positive": len(train_pos), "negative" : len(train_neg) }
    min_sample_size = min(sample_size, key=sample_size.get)
    max_sample_size = max(sample_size, key=sample_size.get)

    min_size = sample_size[min_sample_size]
    max_size = sample_size[max_sample_size]


    if method == 'undersample':
        sample_holder[max_sample_size] = random.sample(sample_holder[max_sample_size], min_size)

    elif method == 'oversample':
        extra = max_size - min_size
        sample_holder[min_sample_size] += random.choices(sample_holder[min_sample_size], k=extra)

    else:
        raise ValueError("Méthode inconnue. Choisir 'undersample' ou 'oversample'.")

    return sample_holder["positive"],sample_holder["negative"]




def get_least_significant_words(n: int, 
                                data: Data) -> list:
    """
    Récupère les mots les moins significatifs en prenant en compte le déséquilibre des classes.
    
    Args:
        n (int): Le nombre de mots à retourner.
        p_m_sachant_pos (Dict[str, float]): Probabilité de chaque mot étant donné que le message est positif.
        p_m_sachant_neg (Dict[str, float]): Probabilité de chaque mot étant donné que le message est négatif.
        prior_pos (float): Probabilité a priori d'un message positif.
        prior_neg (float): Probabilité a priori d'un message négatif.
        vocab (set): L'ensemble des mots uniques dans le corpus.
        
    Returns:
        list: Les n mots les moins significatifs.
    """
    word_probabilities = {}
    if False:
        for word in data.vocab:
            if ( (word in data.p_m_sachant_pos) and (word in data.p_m_sachant_neg) ):
                prob_pos = data.p_m_sachant_pos.get(word, 0) #/ data.nb_pos_word
                prob_neg = data.p_m_sachant_neg.get(word, 0) #/  4 # data.nb_neg_word

                if ( ( prob_pos == 0 ) or ( prob_neg == 0 ) and False ) :# vérification plus trd pour voir si True est plus précis
                    word_probabilities[word] = 1
                    continue

                # Pondérer les probabilités par les priors des classes
                weighted_prob_pos = prob_pos * data.prior_pos
                weighted_prob_neg = prob_neg * data.prior_neg

                # Calculer la différence probabilité pondérée des deux classes
                word_probabilities[word] = abs(weighted_prob_pos - weighted_prob_neg) 
                pass
            pass
        pass
    else:
        word_probabilities = {}

        # Nombre total de documents (messages positifs + négatifs)
        total_docs = data.n_pos + data.n_neg

        # Calcul du TF-IDF pour chaque mot
        for word in data.vocab:
            # TF : fréquence du mot dans chaque classe
            tf_pos = data.p_m_sachant_pos.get(word, 0)
            tf_neg = data.p_m_sachant_neg.get(word, 0)

            # Comptage des documents où le mot apparaît
            doc_count = sum([
                data.count_m_pos[word] if word in data.count_m_pos else 0,
                 data.count_m_neg[word] if word in data.count_m_neg else 0
            ])

            # IDF : Importance inverse du mot dans le corpus
            idf = math.log((total_docs) / (doc_count + 1))  # +1 pour éviter la division par zéro

            # TF-IDF final pour le mot
            word_probabilities[word] = (tf_pos + tf_neg) * idf



    # Trier les mots par probabilité moyenne croissante (les mots les moins significatifs auront une faible probabilité)
    sorted_words = sorted(word_probabilities.items(), key=lambda x: x[1])

    # Extraire les n mots les moins significatifs
    least_significant_words = [word for word, prob in sorted_words[:n]]

    return least_significant_words

#
#
##
#
#"
#
#
#
#
#
#
#
#
#Rédiger la doc pour cette fonction
#
#
#
#
#
#
#
#

def compute_tfidf(data):
    """Calcule les scores TF-IDF pour chaque mot dans le corpus."""
    tfidf_scores = {}
    total_docs = data.n_pos + data.n_neg

    for word in data.vocab:
        tf_pos = data.p_m_sachant_pos.get(word, 0)
        tf_neg = data.p_m_sachant_neg.get(word, 0)
        doc_count = sum([data.count_m_pos[word] if word in data.count_m_pos else 0, data.count_m_neg[word] if word in data.count_m_neg else 0])
        idf = math.log((total_docs) / (doc_count + 1))
        tfidf_scores[word] = (tf_pos + tf_neg) * idf

    return tfidf_scores


#
#
##
#
#"
#
#
#
#
#
#
#
#
#Rédiger la doc pour cette fonction
#
#
#
#
#
#
#
#


def save_word_stats_to_csv(data: Data, filename="word_statistics.csv"):
    """Génère un fichier CSV formaté avec toutes les données jugées utiles pour l'éléboration des stops-words
    le format CSV est utlisé parceque lire ces données dans un tableur Excel est plus confortable et pratique que de les lires dans le débogueur"""
    tfidf_scores = compute_tfidf(data)

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Mot", "Proba positif =", "Probabilité", "Proba négatif =", "Probabilité",
                         "Moyenne pondérée =", "Valeur", "Différence pondérée =", "Valeur", "Score TF-IDF =", "Valeur", "différence du nombre d'occurence pondéré", "valeur"])

        for word in data.vocab:
            prob_pos = data.p_m_sachant_pos.get(word, 0)
            prob_neg = data.p_m_sachant_neg.get(word, 0)
            weighted_avg =( (prob_pos * data.prior_pos) + (prob_neg * data.prior_neg) ) / 2
            weighted_diff = abs(prob_pos * data.prior_pos - prob_neg * data.prior_neg)
            tfidf_score = tfidf_scores.get(word, 0)

            writer.writerow([word, "Proba positif =", f"{prob_pos}", "Proba négatif =", f"{prob_neg}",
                             "Moyenne pondérée =", f"{weighted_avg}", "Différence pondérée =", f"{weighted_diff}",
                             "Score TF-IDF =", f"{tfidf_score}","différence du nombre d'occurence pondéré =",f"{abs( data.count_m_pos.get(word, 0) - (data.count_m_neg.get(word, 0) / 4 ) )}"])

    print(f" Fichier '{filename}' créé avec succès !")
    return


#
#
##
#
#"
#
#
#
#
#
#
#
#
#Rédiger la doc pour cette fonction
#
#
#
#
#
#
#
#


def generate_training_data_with_best_stop_word(n: int, string: list[str], caution: Tableau, rebalancage = False, seed = 55264, mode = "oversample"):

    if rebalancage:
        pos,neg = rebalance_dataset(string,method=mode,seed=seed)
    else:
        pos,neg = seperate_pos_nega(string)
        pass

    conca_pos = conca_str_in_list(pos)
    conca_neg = conca_str_in_list(neg)

    # pos_words_list = [ tokenize( x, caution, type_of_caution_is_default= False ) for x in ( conca_pos.split() ) ]
    # neg_words_list = [ tokenize( x, caution, type_of_caution_is_default= False ) for x in ( conca_neg.split() ) ]

    # pos_final_sorted_dict = liste_occurrences(pos_words_list,caution, type_of_caution_is_default = False)
    # neg_final_sorted_dict = liste_occurrences(neg_words_list,caution, type_of_caution_is_default = False)

    
    caution.function_word_generated = get_least_significant_words(n, generate_training_data(string,caution,caution_type = False) )

    # pos_words_list.clear()
    # neg_words_list.clear()
    # pos_final_sorted_dict.clear()
    # neg_final_sorted_dict.clear()


    pos_words_list = [ tokenize( x, caution, use_default_caution= False ) for x in ( conca_pos.split() ) ]
    neg_words_list = [ tokenize( x, caution, use_default_caution= False ) for x in ( conca_neg.split() ) ]
    pos_final_sorted_dict = liste_occurrences(pos_words_list,caution, use_default_caution = False)
    neg_final_sorted_dict = liste_occurrences(neg_words_list,caution, use_default_caution = False)

    return  ( Data(len(pos), len(neg), pos_final_sorted_dict, neg_final_sorted_dict))
#







    #########################################################
    #########################################################
    #########################################################
    ######### Fonction utilisant le multiprocessing #########
    #########################################################
    #########################################################
    #########################################################

def split_list(lst: list, n: int)->list[list]:
    """
    Divise lst en n sous-listes aussi équilibrées que possible.
    Exemple : split_list([1,2,3,4,5], 2) => [[1,2,3], [4,5]]
    """
    total = len(lst)
    result = []
    start = 0

    for i in range(n):
        # Calcule la taille de cette sous-liste
        remaining_items = total - start
        remaining_slots = n - i
        size = (remaining_items + remaining_slots - 1) // remaining_slots
        end = start + size

        # Découpe et ajoute la sous-liste
        result.append(lst[start:end])
        start = end

        if start >= total:
            break  # Plus rien à découper

    return result

def find_best_stop_word(vocab_chunk, train_df, dev_df, shared_list, lock):
    local_caution = Tableau()
    temp_data_holder = generate_training_data(train_df, local_caution, caution_type=False)
    max_accuracy = naive_bayesian_prediction(dev_df, temp_data_holder, local_caution, mode=2)

    for word in vocab_chunk:
        del temp_data_holder
        local_caution.function_word_generated.append(word)
        temp_data_holder = generate_training_data(train_df, local_caution, caution_type=False)
        acc = naive_bayesian_prediction(dev_df, temp_data_holder, local_caution, mode=2, verbose=False)

        if acc > max_accuracy:
            max_accuracy = acc
        else:
            local_caution.function_word_generated.remove(word)

    # Ajout sécurisé dans la liste partagée
    with lock:
        shared_list.extend(local_caution.function_word_generated)

# ----------------------------
# Fonction exécutée par chaque process
def process_function(seed_subset, max_result, lock, train_df, test_df, caution, tested_seeds_counter, update_queue, timing_info,method = "undersample"):
    for seed in seed_subset:
        start_time = time.time()
        proba_train_recalibrated = generate_training_data_with_calibrage(train_df, caution, random_seed=seed, balancing_method= method)
        a = naive_bayesian_prediction(test_df, proba_train_recalibrated, caution, verbose = False)

        with lock:
            if a > max_result["value"]:
                max_result["seed"] = seed
                max_result["value"] = a

        elapsed = time.time() - start_time
        with lock:
            tested_seeds_counter[0] += 1
            timing_info["total_time"] += elapsed
            timing_info["count"] += 1
            update_queue.put(tested_seeds_counter[0])

        if tested_seeds_counter[0] % 100 == 0:
            print(f"{tested_seeds_counter[0]} seeds testés jusqu'à maintenant")
            pass
        pass
    pass



def worker_non_zero(sublist, tweets, labels, data_proba, caution, lock, best_result):
    for non_zero in sublist:
        prediction = [
            classification_tweet_basique(tweet, data_proba, caution, non_zero_para=non_zero)
            for tweet in tweets
        ]
        accuracy = matrice_confusion(labels, prediction, verbose=True)

        # Mise à jour du meilleur résultat
        with lock:
            if accuracy > best_result['accuracy']:
                best_result['accuracy'] = accuracy
                best_result['non_zero'] = non_zero
                pass
            pass
        pass
    pass

def start_processes2(non_zero_list, test_set, data_proba, caution, num_processes):
    labels,tweets = sep_tweet_label(test_set)
    manager = multiprocessing.Manager()
    best_result = manager.dict({'accuracy': 0.0, 'non_zero': None})
    lock = manager.Lock()

    chunk_size = len(non_zero_list) // num_processes
    processes = []

    for i in range(num_processes):
        start = i * chunk_size
        if (  i  == ( num_processes - 1 ) ):
            end = None  
        else :
            end = (i + 1) * chunk_size
        sublist = non_zero_list[start:end]

        p = multiprocessing.Process(target=worker_non_zero, args=(
            sublist, tweets, labels, data_proba, caution, lock, best_result
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Affichage
    print(f"\n Meilleur non_zero_para : {best_result['non_zero']:.8f} avec une précision de {best_result['accuracy']:.5f}")

    # Sauvegarde dans un fichier texte
    with open("best_non_zero.txt", "w") as f:
        f.write(f"Best non_zero_para: {best_result['non_zero']:.8f}\n")
        f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        return


def start_processes(seed, max_result, lock, train_df, dev_df, caution,
                    tested_seeds_counter, update_queue, timing_info,
                    num_processes, seeds_per_process, method_of_balancing):
    """
    Lance plusieurs processus en parallèle pour tester différentes seeds,
    avec une stratégie de rééchantillonnage spécifiée par l'utilisateur.

    Args:
        seed (list): Liste des seeds à tester.
        max_result (dict): Dictionnaire partagé contenant la meilleure seed et son score.
        lock (multiprocessing.Lock): Verrou pour sécuriser l'accès aux ressources partagées.
        train_df (DataFrame): Jeu d'entraînement.
        dev_df (DataFrame): Jeu de validation.
        caution (Tableau): Instance de la classe Tableau contenant les éléments à retirer.
        tested_seeds_counter (multiprocessing.Value ou Array): Compteur partagé de seeds testées.
        update_queue (multiprocessing.Queue): Queue pour communiquer les mises à jour (UI, logs...).
        timing_info (multiprocessing.Manager().dict): Dictionnaire partagé pour stocker les temps d'exécution.
        num_processes (int): Nombre de processus à lancer.
        seeds_per_process (int): Nombre de seeds à attribuer à chaque processus.
        method_of_balancing (str): Méthode de rééchantillonnage à utiliser ( "oversample" ou "undersample" ).

    Résultat :
        Écrit dans un fichier texte le meilleur résultat trouvé.
    """

    processes = []

    # Création et démarrage des processus
    for i in range(num_processes):
        seed_subset = seed[i * seeds_per_process: (i + 1) * seeds_per_process]# divise la liste des seeds en plusieurs morceaux, chaque process va itérer sur un subset pour trouver le meilleur résultat

        p = multiprocessing.Process(
            target=process_function,
            args=(seed_subset, max_result, lock, train_df, dev_df, caution,
                  tested_seeds_counter, update_queue, timing_info),
            kwargs={"method": method_of_balancing}  # Méthode dynamique passée ici
        )
        processes.append(p)
        p.start()

    # Synchronisation : attendre la fin de tous les processus
    for p in processes:
        p.join()

    # Écriture du résultat final dans un fichier texte
    with open("resultat_final.txt", "w") as f:
        f.write(f"Meilleure seed : {max_result['seed']}\n")
        f.write(f"Accuracy max : {max_result['value']}\n")
        f.write(f"Seeds testés : {tested_seeds_counter[0]}\n")
        pass
    pass

#############################################################
################# Appel de la fonction main #################
#############################################################




if __name__ == "__main__":
    main()
