import os
import torch 
import numpy as np
import pandas as pd 
import torch.nn as nn
import preprocess_data as prep
import torch.nn.functional as F
from scipy.special import inv_boxcox
from scipy.stats import boxcox as fn_boxcox
from loss_func import WISLossFromDistribution
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
THR = 0.01
df_pop_region = pd.read_csv('./data/pop_regional.csv')

df_env = pd.read_csv('data/regional_biome.csv.gz')

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Make sure both are float tensors
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        return torch.mean((torch.log1p(y_pred) - torch.log1p(y_true)) ** 2)
    
class LSTMModel(nn.Module):
    def __init__(self, hidden=8, features=100, predict_n=4, look_back=4, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden
        self.look_back = look_back
        self.features = features

        self.lstm1 = nn.LSTM(
            input_size=features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden, predict_n)

    def gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        # x shape: (batch_size, look_back, features)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take only the output at the last timestep
        x = x[:, -1, :]
        x = self.gelu(x)

        x = self.fc(x)
        x = self.gelu(x)  # mimic final activation from Keras

        return x
    
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(
    model,
    X_train,
    Y_train,
    label,
    batch_size=1,
    epochs=10,
    overwrite=True,
    cross_val=True,
    patience=20,
    monitor='val_loss',
    min_delta=0.0,
    verbose=0,
    doenca='dengue',
    save=True,
    criterion=MSLELoss(),
    lr=0.0005,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, Y_train)

    if cross_val:
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        fold_no = 1
        for train_idx, val_idx in kf.split(dataset):
            print(f'Training fold {fold_no}...')

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        output = model(X_batch)
                        loss = criterion(output, y_batch)
                        val_loss += loss.item()

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            fold_no += 1

    else:
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train, Y_train, test_size=0.25, random_state=7)

        train_loader = DataLoader(TensorDataset(X_train_, y_train_), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val_, y_val_), batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    if save:
        os.makedirs("saved_models", exist_ok=True)
        filename = f"saved_models/trained_{doenca}_{label}.pt"
        if overwrite or not os.path.exists(filename):
            torch.save(model.state_dict(), filename)
            if verbose:
                print(f"Model saved to {filename}")

    return model

def evaluate_samples(model, Xdata, n_passes=100):
    """
    Evaluate a PyTorch LSTM model that outputs (mu, sigma) parameters multiple times.

    Returns:
        mu_preds: numpy array of shape (n_passes, N, predict_n)
        sigma_preds: numpy array of shape (n_passes, N, predict_n)
    """
    device = next(model.parameters()).device
    X_tensor = Xdata.float().to(device)

    # Enable dropout during inference
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()
    model.eval()
    model.apply(enable_dropout)

    predictions = []

    with torch.no_grad():
        for _ in range(n_passes):
            mu, sigma = model(X_tensor)  # (N, predict_n) or (N, 1, predict_n)

            dist = torch.distributions.LogNormal(mu, sigma)
            samples = dist.rsample()  # (N, predict_n)
            
            predictions.append(samples.detach().cpu().numpy())

    # Stack along first dimension: shape (n_passes, N, predict_n)
    predicted = np.stack(predictions, axis=0)

    if predicted.shape[1] == 1:
        predicted = np.squeeze(predicted, axis=1) 

    return predicted

def evaluate(model, Xdata, uncertainty=True, n_passes=100):
    """
    Evaluate a PyTorch LSTM model that outputs (mu, sigma) parameters multiple times.

    Returns:
        mu_preds: numpy array of shape (n_passes, N, predict_n)
        sigma_preds: numpy array of shape (n_passes, N, predict_n)
    """
    model.eval()
    device = next(model.parameters()).device
    X_tensor = Xdata.float().to(device)

    if uncertainty:
        model.train()  # Enable dropout
        mu_preds = []
        sigma_preds = []

        with torch.no_grad():
            for _ in range(n_passes):
                mu, sigma = model(X_tensor)
                mu_preds.append(mu.cpu().numpy())
                sigma_preds.append(sigma.cpu().numpy())

        mu_preds = np.stack(mu_preds, axis=0)      # (n_passes, batch, predict_n)
        sigma_preds = np.stack(sigma_preds, axis=0)

    else:
        with torch.no_grad():
            mu, sigma = model(X_tensor)
        mu_preds = mu.unsqueeze(0).cpu().numpy()      # (1, batch, predict_n)
        sigma_preds = sigma.unsqueeze(0).cpu().numpy()

    return mu_preds, sigma_preds


def sum_regions_predictions(model, df, enso, test_year, columns_to_normalize, boxcox = False, n_passes = 500):
    '''
    Função que aplica o modelo para todas as regionais de saúde e retorna a soma,
    que representa a função para o estado no formato de um dataframe. Não sei se existem formas de
    otimizar esse loop for. 
    '''
    dates = prep.gen_forecast_dates(test_year)

    list_of_enso_indicators = ['enso', 'iod', 'pdo']

    indicators = [item for item in list_of_enso_indicators if item in columns_to_normalize]
 
    predicted = np.zeros((n_passes, 52))

    for geo in df.regional_geocode.unique():

        df_w = prep.aggregate_data(df, geo, column = 'regional_geocode')
            
        if boxcox: 
            df_w['casos']= fn_boxcox(df_w['casos'] + 1, THR)

        #df_w['inc'] = 10*df_w['casos']/df_pop_region.loc[df_pop_region.regional_geocode==geo]['pop'].values[0]
        df_w['pop_norm'] = df_pop_region.loc[df_pop_region.regional_geocode==geo]['pop_norm'].values[0]

        data = df_w.merge(enso[indicators], left_index = True, right_index = True)

        if 'biome' in columns_to_normalize:
            data['biome'] = df_env.loc[df_env.regional_geocode == geo]['biome'].values[0]

        data = data.dropna()

        X_train, y_train, norm_values = prep.get_train_data(data.loc[data.year < test_year], columns_to_normalize= columns_to_normalize)
        X_test, y_test = prep.get_test_data(norm_values, data, test_year, columns_to_normalize)
        
        predicted_ = evaluate_samples(model, X_test, n_passes=n_passes)
        
        predicted_ = predicted_*norm_values['casos']

        if boxcox: 
            predicted_ = inv_boxcox(predicted_.numpy(), THR) -1 

        predicted = predicted + predicted_

    #predicted = predicted.numpy()
    #print(predicted.shape)
    df_preds = pd.DataFrame()

    df_preds['pred'] = np.percentile(predicted, q=50, axis=0)

    df_preds['lower_50'] = np.percentile(predicted, q=25, axis=0)
    df_preds['upper_50'] = np.percentile(predicted, q=75, axis=0)

    df_preds['lower_80'] = np.percentile(predicted, q=10, axis=0)
    df_preds['upper_80'] = np.percentile(predicted, q=90, axis=0)

    df_preds['lower_90'] = np.percentile(predicted, q=5, axis=0)
    df_preds['upper_90'] = np.percentile(predicted, q=95, axis=0)

    df_preds['lower_95'] = np.percentile(predicted, q=2.5, axis=0)
    df_preds['upper_95'] = np.percentile(predicted, q=97.5, axis=0)

    df_preds['date'] = pd.to_datetime(dates)

    return df_preds

class LSTMLogNormalAttentionModel(nn.Module):
    def __init__(self, hidden=8, features=100, predict_n=4, look_back=4, dropout=0.2):
        super(LSTMLogNormalAttentionModel, self).__init__()

        self.hidden_size = hidden
        self.look_back = look_back
        self.features = features
        self.predict_n = predict_n

        self.lstm1 = nn.LSTM(
            input_size=features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # Attention mechanism parameters
        self.attn_weights = nn.Linear(hidden, 1, bias=False)

        # Output layers
        self.fc_mu = nn.Linear(hidden, predict_n)
        self.fc_log_sigma = nn.Linear(hidden, predict_n)

    def gelu(self, x):
        return F.gelu(x)

    def attention_net(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden)
        # Compute attention scores
        attn_scores = self.attn_weights(lstm_output)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum of lstm outputs
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden)
        return context

    def forward(self, x):
        # x shape: (batch_size, look_back, features)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Apply attention over time steps
        x = self.attention_net(x)
        x = self.gelu(x)

        mu = self.fc_mu(x)  # shape: (batch, predict_n)
        log_sigma = self.fc_log_sigma(x)  # shape: (batch, predict_n)
        sigma = torch.exp(log_sigma)

        return mu, sigma
    

class LSTMLogNormalModel(nn.Module):
    def __init__(self, hidden=8, features=100, predict_n=4, look_back=4, dropout=0.2):
        super(LSTMLogNormalModel, self).__init__()

        self.hidden_size = hidden
        self.look_back = look_back
        self.features = features
        self.predict_n = predict_n

        self.lstm1 = nn.LSTM(
            input_size=features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # Output both mu and log_sigma for each predicted value
        self.fc_mu = nn.Linear(hidden, predict_n)
        self.fc_log_sigma = nn.Linear(hidden, predict_n)

    def gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        # x shape: (batch_size, look_back, features)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take only the output at the last timestep
        x = x[:, -1, :]
        x = self.gelu(x)

        mu = self.fc_mu(x)  # shape: (batch, predict_n)
        log_sigma = self.fc_log_sigma(x)  # shape: (batch, predict_n)
        sigma = torch.exp(log_sigma)  # ensure positivity

        return mu, sigma
    
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(
    model,
    X_train,
    Y_train,
    label,
    batch_size=1,
    epochs=10,
    overwrite=True,
    cross_val=True,
    patience=20,
    monitor='val_loss',
    min_delta=0.0,
    verbose=0,
    doenca='dengue',
    save=True,
    #criterion=LogNormalNLLLoss(),
    criterion=WISLossFromDistribution(),
    lr=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, Y_train)

    if cross_val:
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        fold_no = 1
        for train_idx, val_idx in kf.split(dataset):
            print(f'Training fold {fold_no}...')

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    mu, sigma = model(X_batch)
                    loss = criterion(mu, sigma, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        mu, sigma = model(X_batch)
                        loss = criterion(mu, sigma, y_batch)
                        val_loss += loss.item()

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            fold_no += 1

    else:
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train, Y_train, test_size=0.25, random_state=7)

        train_loader = DataLoader(TensorDataset(X_train_, y_train_), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val_, y_val_), batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                mu, sigma = model(X_batch)
                loss = criterion(mu, sigma, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    mu, sigma = model(X_batch)
                    loss = criterion(mu, sigma, y_batch)
                    val_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    if save:
        os.makedirs("saved_models", exist_ok=True)
        filename = f"saved_models/trained_{doenca}_{label}.pt"
        if overwrite or not os.path.exists(filename):
            torch.save(model.state_dict(), filename)
            if verbose:
                print(f"Model saved to {filename}")

    return model
