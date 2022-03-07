#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Importation des librairies 

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import time

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,classification_report,roc_auc_score,precision_score,recall_score, precision_recall_fscore_support 
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn_som.som import SOM
import networkx as nx


pd.set_option('display.max_columns', 100)

from GEM.gem.utils      import graph_util, plot_util
from GEM.gem.evaluation import visualize_embedding as viz
from GEM.gem.evaluation import evaluate_graph_reconstruction as gr
from GEM.gem.embedding.gf       import GraphFactorization
from GEM.gem.embedding.sdne     import SDNE
from argparse import ArgumentParser
from GraphEmbedding.ge import DeepWalk



num_frame=200#arbitraire , a tester plus serieusement


# In[2]:


def create_graph(X_train_ultra_simple):
    g = nx.Graph()

    start = time.time()
    i=0
    while (i<len(X_train_ultra_simple)):
        a=X_train_ultra_simple["merchant"][i]
        b=X_train_ultra_simple["cc_num"][i]
        g.add_edge(a,b,weight=0,edge_id=i)
        i=i+1
    i=0
    while (i<len(X_train_ultra_simple)):# on lis 2 fois mais ca coute que 4 sec
        a=X_train_ultra_simple["merchant"][i]
        b=X_train_ultra_simple["cc_num"][i]
        g[a][b]["weight"]=g[a][b]["weight"]+1 # il falais initialiser en premier
        i=i+1
    
    print("---graph construction = %s seconds ---" % (time.time() - start));start = time.time()
    return g

def fill(g,liste):
  
    while(liste):
        a=liste.pop()
        i=0
        while(i<len(liste)):
            b=liste[i]
            g.add_edge(a,b,weight=1)
            i+=1
    
    #return g

def invert_graph(g):
    new_graph=nx.Graph()
    #tous les arc d'un sommet sont connecté entre eux
    #step 1 = dans new_graph creer un sommet pour chaque arc
    for edge in g.edges:
        new_graph.add_node(g[edge[0]][edge[1]]["edge_id"])
    #pour chaque node de g relier enssemble tous les arc
    for node in g.nodes:
        dico=dict(g[node])# traitement de 1 node
        node_list=[]
        for key in dico:

            node_id=g[node][key]["edge_id"]
            node_list.append(node_id)
        fill(new_graph,node_list)


    return new_graph

def init_sub_graph(nb_frames):
    # divison en plusieures sous graphs 
    sous_graph=[]
    i=0
    i2=0
    sub_g=0
    while ( i<num_frame):
        sous_graph.append(nx.Graph())
        i=i+1
    return sous_graph

def bipartite_dict(dict_merchants,dict_cc_num):

    dict_merchants_copy=dict_merchants.copy()
    dict_merchants_copy = dict([(value, key) for key, value in dict_merchants_copy.items()])
    dict_cc_num_copy=dict_cc_num.copy()
    dict_cc_num_copy = dict([(value, key) for key, value in dict_cc_num_copy.items()])

    for key in dict_merchants_copy.keys():
        dict_merchants_copy[key] = 0
    for key in dict_cc_num_copy.keys():
        dict_cc_num_copy[key] = 1
    return dict_merchants_copy,dict_cc_num_copy

def create_sub_graph(g,nb_frames,dict_merchants,dict_cc_num):
    sous_graph=init_sub_graph(nb_frames)
    dict_merchants_copy,dict_cc_num_copy=bipartite_dict(dict_merchants,dict_cc_num)    
    time_frame_size=len(X_train_ultra_simple) / num_frame
    start = time.time()
    connected_count=0
    i=0
    i2=0
    sub_g=0
    while (i<len(X_train_ultra_simple)):
        a=X_train_ultra_simple["merchant"][i]# rendre plus lisible
        b=X_train_ultra_simple["cc_num"][i]
        if(sous_graph[sub_g].has_edge(a,b)):
            sous_graph[sub_g][a][b]["weight"]=sous_graph[sub_g][a][b]["weight"]+1 
        else:
            sous_graph[sub_g].add_edge(a,b,weight=0)
            #sous_graph[sub_g][a]["bipartite"]=0
            #sous_graph[sub_g][b]["bipartite"]=1
        i=i+1
        i2=i2+1
        if i2>= time_frame_size:
            nx.set_node_attributes(sous_graph[sub_g], dict_merchants_copy, "bipartite")
            nx.set_node_attributes(sous_graph[sub_g], dict_cc_num_copy, "bipartite")
            i2=0
            if(nx.is_connected(sous_graph[sub_g])):
                connected_count+=1
            sub_g=sub_g+1

    print("---graph split = %s seconds ---" % (time.time() - start));start = time.time()

    return sous_graph,connected_count


def create_inverted_sub_graph(g,nb_frames,dict_merchants,dict_cc_num):
    sous_graph,connected_count=create_sub_graph(g,nb_frames,dict_merchants,dict_cc_num)
    inv_sous_graph=[]
    for sg in sous_graph:
        inv_sous_graph.append(invert_graph(sg))
    return inv_sous_graph,connected_count



def start_time_eval():
    start = time.time()
    i=0
    while(i<1000000):
        i=i+1
    boucle_time=time.time() - start
    start = time.time()
    i=0
    while(i<1000000):
        poubelle =time.time()
        i=i+1
    print("---1 milion de time.time=  %s seconds ---" % (time.time() - start-boucle_time));start = time.time()

    


#return le nombre d'arete du graph weighted_g
def nb_edge(weighted_g):
    summ=0
    NODES = list(weighted_g.nodes)
    for node in NODES:
        summ= summ+G.degree[node]
    return summ/2

def replissement(weighted_g, nb_merc,nb_cc_num):
    nb_edges=nb_edge(weighted_g)
    nb_max_edges =(nb_merc* nb_cc_num)#graph bipati
    return  nb_edges / nb_max_edges
    
def edge_repartition(g,len_dict_merchants):
    repartition=[]
    #remplissage de repartiotion avec 0 pour eviter les bugg
    
    #for each vertex 
        # for each edge in vertex.edges
            #repartition[ edge.poid ] ++
    print ("")
    

def slow_concat(d1,d2):
    return dict(d1.items() | d2.items())

def ditc_maping_so_slow_but_why(X_train_ultra_simple,dict_merchants,dict_cc_num):
    #---dictionary maping = 4272.313026428223 seconds ---
    start = time.time()
    size =len(X_train_ultra_simple)
    i=0
    time_val=[]
    while (i<size):
        X_train_ultra_simple.iat[i,0]=dict_merchants[X_train_ultra_simple.iat[i,0]]
        X_train_ultra_simple.iat[i,1]=dict_cc_num[X_train_ultra_simple.iat[i,1]]
        time_val.append(time.time() - start);start = time.time()
        i=i+1
    return time_val


def ditc_maping(X_train_ultra_simple,dict_merchants_cc_num):
    X_train_ultra_simple["merchant"].replace(dict_merchants_cc_num, inplace=True)
    X_train_ultra_simple["cc_num"].replace(dict_merchants_cc_num, inplace=True)

    
def create_split_dict(X_train_ultra_simple):
    start = time.time()
    dict_merchants=dict()
    dict_cc_num=dict()
    index =0
    merc_id=0
    cc_id=0
    while index < len(X_train_ultra_simple):
        if X_train_ultra_simple["merchant"][index] not in dict_merchants.keys():
            dict_merchants[X_train_ultra_simple["merchant"][index]] = merc_id
            merc_id=merc_id+1
        index=index+1

    print("---remplissage dict_merchants  %s seconds ---" % (time.time() - start));start = time.time()
    index=0
    while index < len(X_train_ultra_simple):
        if X_train_ultra_simple["cc_num"][index] not in dict_cc_num.keys():
            dict_cc_num[X_train_ultra_simple["cc_num"][index]] = merc_id
            merc_id=merc_id+1
        index=index+1

    print("---remplissage dict_cc_num %s seconds ---" % (time.time() - start));start = time.time()
    return dict_merchants,dict_cc_num

def create_dict(X_train_ultra_simple):
    dict_merchants,dict_cc_num=create_split_dict(X_train_ultra_simple)
    
    return slow_concat(dict_merchants,dict_cc_num)


def print_info_diverses(X_train_ultra_simple,dico):
    print(X_train_ultra_simple["merchant"][0])
    print(X_train_ultra_simple["cc_num"][0])
    print(X_train_ultra_simple.columns)
    print(X_train_ultra_simple.loc[0]["merchant"])

    print (len(dico) , "humans in the system ")#1676
    print(X_train_ultra_simple["cc_num"][0])
    print (dict_merchants[ "fraud_Rippin, Kub and Mann"], type(dico[ "fraud_Rippin, Kub and Mann"]))
    

#laplacian similarity 1/2
def select_k(spectrum, minimum_energy = 0.9):#
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)

#laplacian similarity 2/2
def laplacian_similarity(graph1,graph2):
    laplacian1 = nx.spectrum.laplacian_spectrum(graph1)
    laplacian2 = nx.spectrum.laplacian_spectrum(graph2)

    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)
    print("k selected =",k)
    similarity = sum((laplacian1[:k] - laplacian2[:k])**2)
    return similarity

def string_edit_dist():
    print("https://anhaidgroup.github.io/py_stringmatching/v0.3.x/Levenshtein.html")
def edit_dist_nx(g1,g2):
    
    for v in nx.optimize_graph_edit_distance(g1, g2):
        minv = v
    return minv


def print_graph_info(connected_count):
    
    print(connected_count,"connected graphs")
    print(len(sous_graph[0].edges)," transactions ")
    print("sous_graph[0][0] , type = ",type(sous_graph[0][0]),"\n")
    print((sous_graph[0][0]))
    print("------------")
    sub_g=0;node=0
    dico=dict(sous_graph[sub_g][node])
    for key in dico:
         print(sous_graph[0][0][key],"key = ",key)
    print(dict(sous_graph[0][0]))
    print("------------")
    print( type(sous_graph[0][0][693]))
    print(sous_graph[0][0][693])
    print("---------")
    print((sous_graph[0].edges))
    
def draw_1(g):
    start = time.time()
    #subax1 = plt.subplot()
    nx.draw(sous_graph[0], with_labels=False, node_size= 1)
    plt.savefig("draw_1.png")
    plt.show()
    print("---draw  %s seconds ---" % (time.time() - start));start = time.time
def draw_2(g):
    start = time.time()
    #subax2 = plt.subplot()
    options = {
        'node_size': 100,
        'width': 3,
    }
    nx.draw_spectral(g, **options)#approximation of the ratio cut
    plt.savefig("draw_2.png")
    plt.show()
    print("---draw  %s seconds ---" % (time.time() - start));start = time.time
def draw_3(g):
    start = time.time()
    #subax3 = plt.subplot()

    nx.draw_shell(g, with_labels=False,node_size= 1)# font_weight='bold')
    plt.savefig("draw_3.png")
    plt.show()
    print("---draw  %s seconds ---" % (time.time() - start));start = time.time
    
def draw_4(g,numb_merchant):
    total=len(g.nodes)#les valeures ascossié ne sont pas les bonnes mais c'est 
    X = list(range(0,numb_merchant ))# juste pour la position geographique 
    Y= list(range(numb_merchant,total ))
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    nx.draw(g, pos=pos,node_size= 1)
    plt.savefig("draw_4.png")
    plt.show()
    
def drawing(g):
    get_ipython().run_line_magic('matplotlib', 'inline')
    draw_1(g)
    draw_2(g)
    draw_3(g)
    draw_4(g,len(dict_merchants.keys()))
    
    plt.show()
    
def complexity_calculations():
    val=0
    train_size=len(Y_train)
    test_size=len(Y_test)
    
    #l'idee premiere
    print("-----------------")
    print("tester la similarité de 1 graph modifié avec tt les autres ")
    print("pour 0.5 sec par calcul et ",num_frame," frames" )
    print( num_frame*0.5," sec")
    print("-----------------")
    print("modifier le graph et recomencer , pour chaques valeures dans train")
    print("pour completement calculer  les similarité de 1 transaction")
    print( num_frame*0.5*train_size," sec")
    print("donc ",int(num_frame*0.5*train_size/(3600*24))," jours")
    print("-----------------")
    
    big_number = num_frame*0.5*train_size*test_size
    big_number_year=int(big_number/31540000)
    print("pour un total de ",big_number," sec")
    print("donc ",big_number_year," ans")
    print("-----------------")
    print("en reduisant la precision au minimum")
    print("chaque transaction n'aura que 1 calcul de similarité")
    print("precision max 50% , doubler le temps de calcul double la precision")
    big_number=0.5*test_size
    print("pour un total de ",big_number," sec")
    print("donc ",int(big_number/3600)," heures")
    
    
    
    


# download data

# In[3]:


program_start = time.time()

import os
data_file= os.path.abspath('../../data')
full_path=data_file+'\\'+'fraudTrain.csv'
train_df=pd.read_csv(full_path)
full_path=data_file+'\\'+'fraudTest.csv'
test_df=pd.read_csv(full_path)

cols = train_df.columns.tolist()
cols = [c for c in cols if c not in ["is_fraud"]]
target = "is_fraud"
print(cols)

#Definition des nouvelles variables X_train and Y_train
X_train = train_df[cols]
Y_train = train_df[target]

#Definition des nouvelles variables X_test and Y_test
X_test = test_df[cols]
Y_test = test_df[target]

features = [ 'merchant', 'cc_num']
X_train = X_train[features]
X_test = X_test[features]

X_train_ultra_simple = X_train.copy()
X_test_ultra_simple = X_test.copy()


# In[4]:


#remplissage des dictionaires

dict_merchants,dict_cc_num=create_split_dict(X_train_ultra_simple)
dictionary=slow_concat(dict_merchants,dict_cc_num)
#dictionary=create_dict(X_train_ultra_simple)


# In[5]:


# associer a chaque marchant son numero dans le dictionaire
#pour la lisibilité , et l'affichage
start = time.time()
ditc_maping(X_train_ultra_simple,dictionary)

print("---dictionary maping = %s seconds ---" % (time.time() - start));start = time.time()


# In[ ]:





# In[6]:


#pip install tk


# 
#     creation du graph
#     

# In[7]:



g = create_graph(X_train_ultra_simple)#40 sec


# In[8]:


#!jupyter notebook --generate-config


# In[9]:


# divison en plusieures sous graphs #20 sec

sous_graph,connected_count=create_sub_graph(g,num_frame,dict_merchants,dict_cc_num)
#sous_graph,connected_count=create_inverted_sub_graph(g,num_frame,dict_merchants,dict_cc_num)


# In[10]:


#pour afficher , attention au cascades
#!jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10


# In[11]:




print(connected_count,"connected graphs")


# In[12]:


i=0
#while(i<num_frame):# doesnt work or my graph are the same
while(i<3):
    start = time.time()

    print ("-------------------------------------")
    #laplacian marche
    val = laplacian_similarity(sous_graph[0],sous_graph[i])
    #val1=nx.graph_edit_distance(sous_graph[0],sous_graph[i],timeout=60,upper_bound=1e10)
    #val2=nx.graph_edit_distance(sous_graph[0],sous_graph[i],timeout=120,upper_bound=1e10)
    #val3=nx.graph_edit_distance(sous_graph[0],sous_graph[i],timeout=180,upper_bound=1e10)
    #print ("comparing ",hex(id(sous_graph[0]))," and ",hex(id(sous_graph[i])))
    #print (val1," ",val2," ",val3) 
    print(sous_graph[0])
    print(sous_graph[i]) 
    print(" similarity = ",val)
    print("---laplacian similarity = %s seconds ---" %         f"{(time.time() - start):.4}"    );start = time.time()

    print ("-------------------------------------")
    i=i+1

#plan,strategie , objectif
#changement de plan
#probleme , le poid des arc devien quoi ? 
#comment ajouter amt apres coup , il suffit de l'ajouter sur les donnée
#avan le SOM
#                                            probably better
#j'inverse le graph                |  je le coupe en n sous_graphs
#je le coupe en n sous_graphs      |  j'inverse les sous_graphs


#je compresse les sous graphs 
#je SOM sur les cous graphs

#complexité= O(n*tps_compression + SOM) // SOM = 6min = negligeable
# In[13]:


def dont_execute():
    """
        ''' Sample usage
    python run_karate.py -node2vec 1
    '''
    parser = ArgumentParser(description='Graph Embedding Experiments on Karate Graph')
    parser.add_argument('-node2vec', '--node2vec',
                        help='whether to run node2vec (default: False)')
    args = vars(parser.parse_args())
    try:
        run_n2v = bool(int(args["node2vec"]))
    except:
        run_n2v = False

    """


# In[14]:


print("---depuis le debut  %s seconds ---" % (time.time() - program_start));start = time.time


# In[15]:


#!git clone https://github.com/shenweichen/GraphEmbedding.git
 
#!cd GraphEmbedding/
#!python setup.py install
#!pip install -U gensim
#!pip install smart_open[all]


# In[16]:


def all_embeding(liste,sous_graph):
    i=0
    num_frame=len(sous_graph)
    G = sous_graph[i]
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=3)
    model.train(window_size=5,iter=3) 
    liste.append(model.get_embeddings())
    i+=1
    while (i<num_frame):

        # train the model and generate embeddings
        G = sous_graph[i]
        model = DeepWalk(G, walk_length=10, num_walks=80, workers=3)
        model.train(window_size=5,iter=3) 
        liste.append(model.get_embeddings())
        i+=1

def test_on_list(liste):
    i=0
    while (i<10):
        liste.append(i)
        i+=1
def int_to_str(G):
    # convert nodes from int to str format
    keys = np.arange(0,int(len(dictionary.keys())))
    values = [str(i) for i in keys]
    dic = dict(zip(keys, values))
    H = nx.relabel_nodes(G, dic)

def my_fun():
    print('How many cats do you have?\n')
    numCats = input()
    try:
        if int(numCats) > 3:
            print('That is a lot of cats.')
        else:
            print('That is not that many cats.')
    except ValueError:
        print("Value error")  
    
    


# In[17]:


embedings=list()
#all_embeding(embedings,sous_graph)


# In[18]:


#all_embeding(embedings,sous_graph)


# In[ ]:


print("try")

def encapsulation(G):
    i=0
    num_frame=len(sous_graph)
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=3)
    model.train(window_size=5,iter=3) 
    return model.get_embeddings()


if(True):
    liste=[]
    i=0
    num_frame=len(sous_graph)
    G = sous_graph[i]
    liste.append(encapsulation(G))
    i+=1

    G = sous_graph[i]
    liste.append(encapsulation(G))
    i+=1


# In[ ]:


#!pip3 install PyQt5


# In[ ]:


# affichage
print(sous_graph[0])
#drawing(sous_graph[0])


# In[ ]:





# In[ ]:


print(len(sous_graph[0].nodes))
print(len(g.nodes))


# In[ ]:



if __name__ ==" __error_code__":# '__main__':

    run_n2v = False

    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    #edge_f = 'data/karate.edgelist'
    edge_f=sous_graph[0].edges()
    # Specify whether the edges are directed
    isDirected = False

    # Load graph

    G = sous_graph[0]

    models = []
    # Load the models you want to run
    models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10**-4, regu=1.0, data_set='karate'))
    #models.append(HOPE(d=4, beta=0.01))
    #models.append(LaplacianEigenmaps(d=2))
    #models.append(LocallyLinearEmbedding(d=2))

    #models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=100,modelfile=['enc_model.json', 'dec_model.json'],weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time.time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        #---------------------------------------------------------------------------------
        print(("\tMAP: {} \t preccision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
        #---------------------------------------------------------------------------------
        # Visualize
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        plt.show()
        plt.clf()


# In[ ]:





# In[ ]:


#370 MiB = 300 000 Mo
# avec g/200 ca compile , tres .... lentement , mais ca compile
too_long=True
if(not too_long):
    val = edit_dist_nx(sous_graph[0],sous_graph[1])


# In[ ]:





# In[ ]:


complexity_calculations()


# In[ ]:


drawing(g)


# In[ ]:


print( type(sous_graph[0][1][694]))


# In[ ]:


print_graph_info(connected_count)


# In[ ]:





# In[ ]:




