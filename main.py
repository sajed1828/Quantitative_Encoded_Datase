import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from alpha_core import alpha_factor
from alpha_zero_101 import Alpha_Zero
from modul_features_map import Autoencoder, genlenDataset, train_autoencoder, encode_data
from data_resorces import data_source
import os
import pandas as pd


url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url)

df_sp500 = table[0]
symbols = df_sp500["Symbol"].tolist()

# main script
if __name__ == "__main__":
    now = datetime.utcnow()
    start_date = now - timedelta(days=365)
    
    symbols = [s.replace('.', '_') for s in symbols]
    symbols = [s for s in symbols if s.isalpha() or '_' in s]
    symbols = symbols[:2]
    print(symbols)
    df_all = []
    for symbol in symbols:
      try:  
        print(f"\n📥 Processing symbol: {symbol}")
        ds = data_source(
            symbols=[symbol],  
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
            freq="5min",
            delay=1.5,
            save_csv=True)

        df = ds.run()
        if df.empty:
            print(f"\b⚠️ No data fetched for {symbol}. Skipping.")
            continue

        print("✅ Raw Sample:\n", df.head())

        df_sym = alpha_factor.ta_factor_indcators(df)
        df_sym = Alpha_Zero.colpute_all_alpha_factors(df_sym)
        
        df_path = "df_data"
        os.makedirs(df_path, exist_ok=True)
        save_df_path = os.path.join(df_path, f"encoded_data_{symbol}.csv")
        df_sym.to_csv(save_df_path, index=False)

        FEATURE_COLUMNS = df_sym.select_dtypes(include=('float64', 'uint64')).columns.tolist()
        FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if df_sym[col].notna().all()]

        if len(FEATURE_COLUMNS) < 5:
            print(f"⚠️ Not enough features for {symbol}, skipping.")
            continue

        print(f"✅ Features ({len(FEATURE_COLUMNS)}):", FEATURE_COLUMNS)

        dataset = genlenDataset(df=df_sym, FEATURE_COLUMNS=FEATURE_COLUMNS, qen_len=31)
        if len(dataset) == 0:
            print(f"⚠️ Not enough data for training for {symbol}, skipping.")
            continue

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        sample_x, _ = dataset[0]
        input_dim = sample_x.numel()

        model = Autoencoder(input_dim=input_dim, encoding_dim=10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_autoencoder(dataloader, model, optimizer, num_epochs=10)

        encoded_data = encode_data(dataloader, model)
        encoded_df = pd.DataFrame(encoded_data, columns=[f"feature_{i+1}" for i in range(encoded_data.shape[1])])
        encoded_df["symbol"] = symbol
        
        output_dir = "encoded_data"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"encoded_data_{symbol}.csv")
        encoded_df.to_csv(save_path, index=False)
        
        df_all.append(encoded_df)
      
      except:
          print(f"Download_{symbol}_is_faild")

    concet_all = "concet_database"
    os.makedirs(concet_all, exist_ok=True)
    fienl_path = os.path.join(concet_all, 'concat_all_data_00.csv')
    df_fienl = pd.concat(df_all, ignore_index=False)
    df_fienl.to_csv(fienl_path, index=False)

