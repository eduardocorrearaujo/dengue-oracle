import warnings
import pandas as pd
from itertools import product
import preprocess_data as prep
from models import LSTMLogNormalModel, sum_regions_predictions
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import torch 

df_pop_region = pd.read_csv('./data/pop_regional.csv')
THR = 0.1

states =  ['SC', 'PR', 'RS', 'SP', 'MG', 'RJ', 'ES', 'BA', 'CE', 'PE', 'PB', 'PI', 'RN', 
               'MA', 'AL', 'SE', 'DF', 'MT', 'MS', 'GO','RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO']

estado_para_regiao = {  'SC': 'Sul',
                            'PR': 'Sul',
                            'RS': 'Sul',
                            'SP': 'Sudeste',
                            'MG': 'Sudeste',
                            'RJ': 'Sudeste',
                            'ES': 'Sudeste',
                            'BA': 'Nordeste',
                            'CE': 'Nordeste',
                            'PE': 'Nordeste',
                            'PB': 'Nordeste',
                            'PI': 'Nordeste',
                            'RN': 'Nordeste',
                            'MA': 'Nordeste',
                            'AL': 'Nordeste',
                            'SE': 'Nordeste',
                            'DF': 'Centro - Oeste',
                            'MT': 'Centro - Oeste',
                            'MS': 'Centro - Oeste',
                            'GO': 'Centro - Oeste',
                            'RO': 'Norte',
                            'AC': 'Norte',
                            'AM': 'Norte',
                            'RR': 'Norte',
                            'PA': 'Norte',
                            'AP': 'Norte',
                            'TO': 'Norte'}

if __name__ == '__main__': 
    model_name = 'covar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxcox = False 

    for state, TEST_YEAR in product(states, [2023, 2024,2025]): 
        print(f'{state} - {TEST_YEAR}')
        df = prep.load_cases_data()
        enso = prep.load_enso_data()

        df = df.loc[df.uf == state]

        
        # base model 
        if model_name == 'lognorm':
            model = LSTMLogNormalModel(hidden=64, features=3, 
                            predict_n=52, look_back=89)
            
            columns_to_normalize = ['casos','epiweek']
                    
        if model_name == 'covar': 
            model = LSTMLogNormalModel(hidden=64, features=5, 
                            predict_n=52, look_back=89)

            columns_to_normalize = ['casos','epiweek', 'biome', 'enso']

                    
        model_path = f'./saved_models/trained_dengue_{estado_para_regiao[state]}_{TEST_YEAR-1}_{model_name}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)  

        df_preds = sum_regions_predictions(model, df, enso, TEST_YEAR, columns_to_normalize, boxcox=boxcox)

        df_preds.to_csv(f'predictions/preds_region_{model_name}_{state}_{TEST_YEAR}.csv', index = False)
