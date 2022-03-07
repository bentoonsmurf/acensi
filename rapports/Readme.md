# Projet Prévention Fraude CB
***
Aujourd'hui, nous avons beaucoup de demande de confirmation lors d'un achat en ligne, le but étant d'assurer un maximum de sécurité, de confort et de sérénité lors de nos achats en ligne.  
Avec ce projet, nous désirons obtenir les mêmes effets tout en diminuant drastiquement  ces confirmations.  
Pour ce faire, nous avons mis en place un site web pour présenter plusieurs approches de prévention, une détection et  nous allons plus loin avec la théorie des jeux pour devancer les futures menaces.  
Lien [GitLab](https://gitlab.com/finance.acensi/fraude) : https://gitlab.com/finance.acensi/fraude
***
![Image text](https://www.acensi.fr/bundles/pageoverride/img/logo-acensi-bleu.svg)
***
## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)
4. [collaboration](#collaboration)
5. [FAQs](#faqs)

## General Info
***
### Statut du projet : ( en cours )
__Site web :__
* Home Page : à faire
* Analyse exploratoire : ok ( peut être refaire la présentation)
* Modèles : 
  * Algorithmes : ok ( peut être refaire la présentation)
  * Comparaison des Algorithmes : ok (peut être changer d'emplacement)
  * Théorie des jeux : en cours
  * (images, model, présentations, etc...)
* Détection (Prédiction) : ok (refaire la présentation)
  * New clients
  * New marchands
  * Détection
    * (texte plus compréhensible, séparer  client et marchand dans une liste d'onglets (cf: modèles))
    * (enlever "timeunix" et utiliser les dates + heure pour l'insérer dans le formulaire)
* Admin : ok (intéressant tant que le site est en fabrication, peut être l'enlever ensuite)
## Technologies
***
Liste des technologies utilisées avec
[python](https://www.python.org/doc/) pour le projet :

(pour avoir la totalité des technologies, veuillez vous référer au fichier **requirements.txt**)
* __Machine Learning__
  * [Pandas](https://pandas.pydata.org/docs/) : Version 1.3.5
  * [Seaborn](https://seaborn.pydata.org/) : Version 0.11.2
  * [NumPy](https://numpy.org/doc/) : Version 1.21.4
  * [Scikit-kit learn](https://scikit-learn.org/stable/) : Version 0.24.2 (important !)
  * [Shap](https://shap.readthedocs.io/en/latest/index.html) : Version ???
  * [Xgboost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html) : Version 1.3.0rc1 (important !)
  * [Tensorflow-keras](https://www.tensorflow.org/api_docs) : Version 2.7.0
* __Game Theory :__
  * [Cvxopt](http://cvxopt.org/documentation/) : Version ???
* __Web__ :
  * [Django](https://docs.djangoproject.com/en/4.0/) : Version 4.0
## Installation
***
Pour faire fonctionner la plateforme web de Fraude, il faudra effectuer plusieurs étapes.  
Pour plus de facilité, nous n'utiliserons que le logiciel *Git CMD* pour tout ce qui va suivre.  
Installez __[Git (version windows 64bits)](https://github.com/git-for-windows/git/releases/download/v2.34.1.windows.1/Git-2.34.1-64-bit.exe)__ en cliquant sur le lien.  
Une fois l'installation faite, veuillez utiliser le __Git CMD__ pour la suite :
  
* créer un dossier projetFraude :
```
mkdir projetFraude && cd projetFraude
```
* configurer le Git :
```
git config --global user.name "Guirado Léo"
```
Veuillez utiliser votre Nom Prénom
```
git config --global user.email "leo.guirado@acensi.fr"
```
Veuillez utiliser votre email
```
cd C:\Users\leo.guirado\projetFraude
```
Veuillez modifier le chemin d'accès pour aller dans votre dossier projetFraude
```
git init &&
git status
```
```
git remote add origin https://gitlab.com/finance.acensi/fraude.git
```
* faire un clone :
```
git clone git@gitlab.com:finance.acensi/fraude.git
```
* installer [python 3.9](https://www.python.org/downloads/release/python-390/) (ou plus récent du moment qu'il n'est __PAS trop__ récent)
* installer un environnement virtuel :
```
cd C:\Users\leo.guirado\projetFraude
```
Veuillez modifier le chemin d'accès pour aller dans votre dossier projetFraude
```
py -m venv .env &&
cd .env/scripts &&
activate
```
(s'il y a une erreur avec activate, faire la commande suivante)
```
./activate
```
(un "(.env)" doit apparaître avant le chemin d'accès)

* installer les librairies spécifiés et leur version demandée :
```
cd C:\Users\leo.guirado\projetFraude\WEB
```
Veuillez modifier le chemin d'accès pour aller dans votre dossier projetFraude\WEB
```
pip install -r requirements.txt
```
* mise en route du server (toujours dans le dossier WEB) :
```
py manage.py runserver
```
## Collaboration
***
#### Ce projet à été mis en place par Mr James KOUTHON.
La partie Data Scientist à été faite par :  
* Anas SORY  

La partie Game Theory à été faite par :  
* EL BOURI Mohamed Ali  
  
La partie Graph Theory à été faite par :  
* MARTY Benjamin  
  
La partie Web à été faite par :  
* GUIRADO Léo  
## FAQs
***
* Toute installation de "librairies" doivent se faire dans le dossier WEB lorsque l'environnement virtuel est activé, le tout via l'invite de commande.
***
* S'il y a une modification des librairies, merci de mettre à jour le fichier __requirements.txt__. Il est possible de le faire via l'invite de commande :
```
pip freeze > requirements.txt
```
***
* Lors de la mise en place du site sur internet, aller dans settings.py et changer 
```
DEBUG = True
``` 
par 
```
DEBUG = False
```
***
* Désactiver l'environnement virtuel (dans l'invite de commande) :
```
cd .env/scripts &&
deactivate
```
(s'il y a une erreur avec deactivate, faire la commande suivante)

```
./deactivate
```
***
* Attention au versions utilisées.  
Par defaut les installeur installent la version la plus a jours.  
Des versions trop recentes ou anciennes  peuvent rendre le code non fonctionnel.  
Les versions mentionées (important !) doivent rester a cette version là
***
* les mots en bleu dans le Readme.md sont des liens, n'hésitez pas à les utiliser.