import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.visuals import plot_confusion_matrix, plot_pr_curve
from src.utils.preprocessing import create_sequences, add_gaussian_noise
from src.utils.analysis import run_parameter_analysis
from src.models.automata import ProbabilisticAutomata
from src.models.dl_models import build_lstm_model, train_dl_model, build_gru_model

# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def predict_automata_sequence(automata_model, oruntuler_test, threshold):
    preds = [0] 
    for i in range(len(oruntuler_test) - 1):
        mevcut = oruntuler_test[i]
        gelen = oruntuler_test[i+1]
        karar = automata_model.explain_decision(i+1, mevcut, gelen, threshold)
        preds.append(1 if karar["decision"] == "anomaly" else 0)
    return preds

def print_metrics(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"[{model_name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

# ==========================================
# KONFİGÜRASYON VE VERİ YÜKLEME
# ==========================================
with open("configs/config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

batadal_tam_veri = pd.read_csv(config["paths"]["batadal"])
batadal_tam_veri.columns = batadal_tam_veri.columns.str.strip()

hedef_sutun = "ATT_FLAG"
zaman_sutunu = "DATETIME"
ozellikler = [col for col in batadal_tam_veri.columns if col not in [hedef_sutun, zaman_sutunu]]

toplam_satir = len(batadal_tam_veri)
train_siniri = int(toplam_satir * config["split_ratios"]["train"])
val_siniri = train_siniri + int(toplam_satir * config["split_ratios"]["validation"])

train_df = batadal_tam_veri.iloc[:train_siniri].copy()
val_df = batadal_tam_veri.iloc[train_siniri:val_siniri].copy()
test_df = batadal_tam_veri.iloc[val_siniri:].copy()

scaler = StandardScaler()
train_df[ozellikler] = scaler.fit_transform(train_df[ozellikler])
val_df[ozellikler] = scaler.transform(val_df[ozellikler])
test_df[ozellikler] = scaler.transform(test_df[ozellikler])

# PCA (Automata için Tek Boyut)
pca = PCA(n_components=1)
train_df["PC1"] = pca.fit_transform(train_df[ozellikler])
val_df["PC1"] = pca.transform(val_df[ozellikler])
test_df["PC1"] = pca.transform(test_df[ozellikler])

print(f"Train Verisi Satır Sayısı: {train_df.shape[0]}")
print(f"Validation Verisi Satır Sayısı: {val_df.shape[0]}")
print(f"Test Verisi Satır Sayısı: {test_df.shape[0]}")

# ==========================================
# AUTOMATA EĞİTİMİ VE DİNAMİK THRESHOLD
# ==========================================
w_size = config["automata_params"]["window_size"]
a_size = config["automata_params"]["alphabet_size"]

automata = ProbabilisticAutomata(window_size=w_size, alphabet_size=a_size)
train_pc1 = train_df["PC1"].values
sax_dizisi_train = automata.transform_to_sax(train_pc1)
oruntuler_train = automata.extract_patterns(sax_dizisi_train)
automata.fit(oruntuler_train)

# Dinamik Threshold (Yüzdelik Dilim - %10)
train_probs = []
for current_state, next_states in automata.transitions.items():
    for next_state in next_states:
        train_probs.append(automata.get_transition_probability(current_state, next_state))

dinamik_threshold = np.percentile(train_probs, 10) if train_probs else 0.05
print(f"\n[*] Automata %10'luk Dinamik Threshold (BATADAL): {dinamik_threshold:.4f}")

# ==========================================
# DL VE AUTOMATA ORTAK TEST VERİSİ HAZIRLIĞI
# ==========================================
time_steps = config["automata_params"]["window_size"]

X_train_dl = train_df[ozellikler].values
y_train_raw = batadal_tam_veri.loc[train_df.index, hedef_sutun].astype(float).values
y_train_dl = np.where(y_train_raw > 0, 1, 0)

X_val_dl = val_df[ozellikler].values
y_val_raw = batadal_tam_veri.loc[val_df.index, hedef_sutun].astype(float).values
y_val_dl = np.where(y_val_raw > 0, 1, 0)

X_test_dl = test_df[ozellikler].values
y_test_raw = batadal_tam_veri.loc[test_df.index, hedef_sutun].astype(float).values
y_test_dl = np.where(y_test_raw > 0, 1, 0)

X_train_seq, y_train_seq = create_sequences(X_train_dl, y_train_dl, time_steps)
X_val_seq, y_val_seq = create_sequences(X_val_dl, y_val_dl, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_dl, y_test_dl, time_steps)

# ==========================================
# AUTOMATA GERÇEK TEST (INFERENCE)
# ==========================================
test_pc1 = test_df["PC1"].values
sax_dizisi_test = automata.transform_to_sax(test_pc1)
oruntuler_test = automata.extract_patterns(sax_dizisi_test)

automata_preds = predict_automata_sequence(automata, oruntuler_test, threshold=dinamik_threshold)

# Hizalama ve Skor
min_len = min(len(y_test_seq), len(automata_preds))
automata_preds_hizali = automata_preds[-min_len:] if len(automata_preds) > min_len else automata_preds

f1_auto_batadal = f1_score(y_test_seq[:min_len], automata_preds_hizali[:min_len], zero_division=0)
print(f"[*] Automata Gerçek Test F1-Score (BATADAL): {f1_auto_batadal:.4f}")

# Hoca İçin Gerçek Veriden JSON Çıktısı (Test Seti Time Step 6)
ornek_mevcut = oruntuler_test[5]
ornek_gelen = oruntuler_test[6]
aciklama_json = automata.explain_decision(time_step=6, current_state=ornek_mevcut, incoming_pattern=ornek_gelen, threshold=dinamik_threshold)

print("\n[SYSTEM DECISION - GERÇEK VERİ İLE JSON FORMATI]")
print(json.dumps(aciklama_json, indent=4))

# ==========================================
# DERİN ÖĞRENME (LSTM & GRU) EĞİTİMLERİ
# ==========================================
siniflar = np.unique(y_train_seq)
agirliklar = compute_class_weight(class_weight='balanced', classes=siniflar, y=y_train_seq)
sinif_agirliklari = dict(zip(siniflar, agirliklar))
print(f"\nUygulanan Sınıf Ağırlıkları: {sinif_agirliklari}")

epochs = config["training_params"]["epochs"]
batch_size = config["training_params"]["batch_size"]
patience = config["training_params"]["early_stopping_patience"]
seed = config["training_params"]["random_seeds"][0]

lstm_model = build_lstm_model((time_steps, len(ozellikler)))
print("\n--- LSTM Modeli Eğitimi Başlıyor ---")
history_lstm = train_dl_model(
    lstm_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
    epochs=epochs, batch_size=batch_size, patience=patience, seed=seed, class_weight=sinif_agirliklari
)

gru_model = build_gru_model((time_steps, len(ozellikler)))
print("\n--- GRU Modeli Eğitimi Başlıyor ---")
history_gru = train_dl_model(
    gru_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
    epochs=epochs, batch_size=batch_size, patience=patience, seed=seed, class_weight=sinif_agirliklari
)

# ==========================================
# İSTATİSTİKSEL DENEY (5 SEED)
# ==========================================
print("\n--- 5 Farklı Seed ile İstatistiksel Deney Başlıyor ---")
seeds = config["training_params"]["random_seeds"]
lstm_f1_scores = []
gru_f1_scores = []

for s in seeds:
    print(f"\n>>> Seed: {s} için eğitim yapılıyor...")
    
    lstm = build_lstm_model((time_steps, len(ozellikler)))
    train_dl_model(lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs, batch_size=batch_size, patience=patience, seed=s, class_weight=sinif_agirliklari)
    lstm_preds = np.where(lstm.predict(X_test_seq, verbose=0) > 0.5, 1, 0)
    lstm_f1_scores.append(f1_score(y_test_seq, lstm_preds, zero_division=0))
    
    gru = build_gru_model((time_steps, len(ozellikler)))
    train_dl_model(gru, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs, batch_size=batch_size, patience=patience, seed=s, class_weight=sinif_agirliklari)
    gru_preds = np.where(gru.predict(X_test_seq, verbose=0) > 0.5, 1, 0)
    gru_f1_scores.append(f1_score(y_test_seq, gru_preds, zero_division=0))

print("\n=== DENEY SONUÇLARI (BATADAL) ===")
print(f"LSTM F1-Score Ortalama: {np.mean(lstm_f1_scores):.4f} (± {np.std(lstm_f1_scores):.4f})")
print(f"GRU F1-Score Ortalama: {np.mean(gru_f1_scores):.4f} (± {np.std(gru_f1_scores):.4f})")
print(f"Automata F1-Score (Tek Test): {f1_auto_batadal:.4f}")

# ==========================================
# AUTOMATA PARAMETRE ANALİZİ
# ==========================================
print("\n--- Otomata Parametre Analizi Başlıyor ---")
w_varyasyonlari = config["automata_params"].get("window_variations", [3, 4, 5, 6])
a_varyasyonlari = config["automata_params"].get("alphabet_variations", [3, 4, 5, 6])

analiz_sonuclari = run_parameter_analysis(train_df["PC1"].values, w_varyasyonlari, a_varyasyonlari)
print("\n[PARAMETRE ANALİZ TABLOSU]")
print(analiz_sonuclari.to_string(index=False))

# ==========================================
# GÜRÜLTÜ (ROBUSTNESS) ANALİZİ
# ==========================================
print("\n--- Gürültülü Veri (Gaussian Noise) Senaryosu Başlıyor ---")
X_test_noisy = add_gaussian_noise(X_test_dl, mean=0.0, std=0.05)
X_test_noisy_seq, _ = create_sequences(X_test_noisy, y_test_dl, time_steps)

lstm_preds_prob_noisy = lstm_model.predict(X_test_noisy_seq, verbose=0)
lstm_preds_noisy = np.where(lstm_preds_prob_noisy > 0.5, 1, 0)

lstm_preds_prob_orijinal = lstm_model.predict(X_test_seq, verbose=0)
lstm_preds_orijinal = np.where(lstm_preds_prob_orijinal > 0.5, 1, 0)

print("Orijinal Test Verisi Performansı:")
print_metrics("LSTM - Original", y_test_seq, lstm_preds_orijinal)

print("\nGürültü Eklenmiş (Noisy) Test Verisi Performansı:")
print_metrics("LSTM - Noisy", y_test_seq, lstm_preds_noisy)