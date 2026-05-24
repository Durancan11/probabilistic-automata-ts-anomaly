import json
import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from src.utils.preprocessing import create_sequences
from src.models.dl_models import build_lstm_model, train_dl_model

with open("configs/config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

skab_veri_parcalari = []
skab_klasorleri = [config["paths"]["skab_valve1"], config["paths"]["skab_valve2"]]

for klasor in skab_klasorleri:
    klasor_adi = os.path.basename(klasor)
    csv_dosyalari = glob.glob(os.path.join(klasor, "*.csv"))
    
    for dosya_yolu in csv_dosyalari:
        dosya_adi = os.path.basename(dosya_yolu)
        df = pd.read_csv(dosya_yolu, sep=";")
        df["source_group"] = klasor_adi
        df["source_file"] = dosya_adi
        skab_veri_parcalari.append(df)

skab_tam_veri = pd.concat(skab_veri_parcalari, ignore_index=True)

hedef_sutun = "anomaly"
haric_tutulacaklar = ["datetime", "changepoint", "source_group", "source_file", hedef_sutun]
ozellikler = [col for col in skab_tam_veri.columns if col not in haric_tutulacaklar]

gkf = GroupKFold(n_splits=3)
gruplar = skab_tam_veri["source_file"].values
X_skab = skab_tam_veri[ozellikler].values
y_skab = skab_tam_veri[hedef_sutun].values

time_steps = config["automata_params"]["window_size"]
epochs = config["training_params"]["epochs"]
batch_size = config["training_params"]["batch_size"]
patience = config["training_params"]["early_stopping_patience"]
seed = config["training_params"]["random_seeds"][0]

fold_no = 1
lstm_fold_f1 = []

print("\n--- SKAB Veri Seti Üzerinde GroupKFold Analizi Başlıyor ---")

for train_idx, test_idx in gkf.split(X_skab, y_skab, gruplar):
    print(f"\n>>> Fold {fold_no} Eğitim ve Testi <<<")
    
    X_train, X_test = X_skab[train_idx], X_skab[test_idx]
    y_train, y_test = y_skab[train_idx], y_skab[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)
    
    siniflar = np.unique(y_train_seq)
    from sklearn.utils.class_weight import compute_class_weight
    agirliklar = compute_class_weight(class_weight='balanced', classes=siniflar, y=y_train_seq)
    sinif_agirliklari = dict(zip(siniflar, agirliklar))
    
    lstm = build_lstm_model((time_steps, len(ozellikler)))
    train_dl_model(
        lstm, X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
        epochs=epochs, batch_size=batch_size, patience=patience, seed=seed, class_weight=sinif_agirliklari
    )
    
    lstm_preds = np.where(lstm.predict(X_test_seq, verbose=0) > 0.5, 1, 0)
    f1 = f1_score(y_test_seq, lstm_preds, zero_division=0)
    lstm_fold_f1.append(f1)
    
    print(f"Fold {fold_no} Test F1-Score: {f1:.4f}")
    fold_no += 1

print("\n=== SKAB DENEY SONUÇLARI (GroupKFold) ===")
print(f"LSTM Ortalama F1-Score: {np.mean(lstm_fold_f1):.4f} (± {np.std(lstm_fold_f1):.4f})")