import pandas as pd

def count_values():
    '''
    Entenda o desbalanceamento antes de modelar:
    - Antes de qualquer coisa:
    - Analise a distribuição das classes
    - Identifique classe majoritária e minoritária
    - Avalie a criticidade do erro (ex: falso negativo)
    '''
    
    return df['label'].value_counts(normalize=True)
