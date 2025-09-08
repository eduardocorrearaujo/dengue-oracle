import os
import torch 
import pandas as pd
from epiweeks import Week
from itertools import product
import preprocess_data as prep
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
from models import LSTMLogNormalModel, train 

'''
This script is used to train the model for a specific STATE and forecast the cases on a 
specific year (TEST_YEAR). The model is trained with the regional health data before the year selected. 
'''
    
if __name__ == '__main__':

    model_name = 'covar'

    if model_name == 'lognorm':

        columns_to_normalize = ['casos','epiweek']
                    
    if model_name == 'covar': 

        columns_to_normalize = ['casos','epiweek', 'biome', 'enso']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    regioes_estados = {
        'Sul': ['SC', 'PR', 'RS'],
        'Sudeste': ['SP', 'MG', 'RJ', 'ES'],
        'Nordeste': ['BA', 'CE', 'PE', 'PB', 'PI', 'RN', 'MA', 'AL', 'SE'],
        'Centro - Oeste': ['DF', 'MT', 'MS', 'GO'],
        'Norte': ['RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO']
    } 
    
    states =  ['SC', 'PR', 'RS', 'SP', 'MG', 'RJ', 'ES', 'BA', 'CE', 'PE', 'PB', 'PI', 'RN', 
               'MA', 'AL', 'SE', 'DF', 'MT', 'MS', 'GO','RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO']
    
    boxcox = False
    
    for region, TEST_YEAR in product(regioes_estados.keys(), [2023, 2024,2025]): 

        print(f'{region} - {TEST_YEAR}')
        df = prep.load_cases_data()
        
        df = df.loc[df.uf.isin(regioes_estados[region])]
        df = df.loc[df.index >= pd.to_datetime(Week(2015,41).startdate())]
        enso = prep.load_enso_data()

        # generate the samples to train and test based on the regional data 
        X_train, y_train = prep.generate_regional_train_samples(df, enso, TEST_YEAR, columns_to_normalize=['casos','epiweek', 'biome', 'enso'], boxcox = boxcox)

        model = LSTMLogNormalModel(hidden=64, features=len(columns_to_normalize) + 1, 
                        predict_n=52, look_back=89)
            
        label = f'{region}_{TEST_YEAR-1}_{model_name}'
        batch_size = 1
        epochs = 200
        cross_val = False
        verbose = 0
        doenca = 'dengue'
        min_delta = 0
        patience= 30

        if TEST_YEAR > 2023:     
            model_path = f'./saved_models/trained_dengue_{region}_{TEST_YEAR-2}_{model_name}.pt'
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)  
    
        model = train(model, X_train, y_train, label=label, batch_size=batch_size, epochs=epochs,
                                                    overwrite=True, cross_val = cross_val, monitor='val_loss',
                                                    verbose=verbose, doenca=doenca,
                                                    min_delta = min_delta, patience=patience)