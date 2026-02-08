from imblearn.over_sampling import SMOTE

def resampling_values():
    '''
    Reamostragem dos dados
    Oversampling (classe minoritária)
    - Random Oversampling
    - SMOTE
    - ADASYN
    '''

    X_res, y_res = SMOTE().fit_resample(X, y)

