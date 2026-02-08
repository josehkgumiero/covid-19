from sklearn.ensemble import RandomForestClassifier

def scikit_class_weight():
    '''
    Use pesos de classe (Class Weights)
    Muito eficaz quando não dá para alterar o dataset.
    Exemplo (Scikit-learn)
    '''
    model = RandomForestClassifier(class_weight='balanced')

def keras_class_weight():
    '''
    Use pesos de classe (Class Weights)
    Muito eficaz quando não dá para alterar o dataset.
    Exemplo (Deep Learning – Keras)
    '''
    class_weight = {0: 6, 1: 1, 2: 1}

