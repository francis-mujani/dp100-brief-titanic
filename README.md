![dp100 image](img/dp100.png)

# brief-titanic

Réaliser un projet complet de data science avec la plateforme azure sur les données du titanic :
- hébérgement des données
- préparation
- entraintement d'un modèle
- prédiction
- déploiement de la solution 

## Livrables

Un repo git qui contient un notebook bien structuré qui réalise les étapes demandées + une url qui permette de réaliser des prédictions.

## Référentiels

dp100

## Contexte du projet

- Uploadez les données contenues dans le fichier [titanic.csv](https://github.com/jtobelem-simplon/dp100-brief-titanic/blob/master/data/titanic.csv) dans un dataset référencé sur votre workspace
- Utilisez le fichier [titanic_training.py](https://github.com/jtobelem-simplon/dp100-brief-titanic/blob/master/script/titanic_training.py) pour entrainer un modèle d'après ce dataset
- Déployez votre solution 


## Modalités pédagogiques

Individuel en autonomie.

## Critère de performance

*(depuis la plateforme azure)*
- connectez-vous dans votre workspace sur le studio azure
- demarrez votre votre machine
- ouvrez jupyter notebook et lancez un "new" terminal
- clonez https://github.com/jtobelem-simplon/dp100-brief-titanic dans votre dossier
- ouvrez le notebook [titanic-azure.ipynb](https://github.com/jtobelem-simplon/dp100-brief-titanic/blob/master/titanic-azure.ipynb)

*(depuis un environnement local)*
- clonez https://github.com/jtobelem-simplon/dp100-brief-titanic dans votre dossier
- téléchargez dedans le fichier config.json depuis le portail azure
- activez l'environnement "azure"
- démarrez jupyter notebook dans cet environnement
- ouvrez le notebook [titanic-azure.ipynb](https://github.com/jtobelem-simplon/dp100-brief-titanic/blob/master/titanic-azure.ipynb)

---

*(suite pour tous)*
- vérifiez que les cellules de la partie "1 Configuration" fonctionnent
- uploadez les données dans un tabular dataset et référencez (register) ce dataset dans votre workspace
- créez un estimator qui prenne en "inputs" le dataset
- l'accuracy et la mae (mean absolute error) seront loggés à chaque run
- exécutez cette estimator dans une expérience

> sur la plateforme azure, observez les résultats du run

- le modèle produit sera enregistré dans un fichier pkl et référencé dans le workspace
- faites plusieurs run avec des models différents
- déployez votre solution et faites un test avec postman


## Modalités d'évaluation

Sur votre poste en montrant à votre formateur votre niveau d'avancement sur les modules d'apprentissage.
