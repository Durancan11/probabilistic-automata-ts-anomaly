import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

pca = PCA(n_components=1)
train_pca = pca.fit_transform(train_df[ozellikler])
val_pca = pca.transform(val_df[ozellikler])
test_pca = pca.transform(test_df[ozellikler])

train_df["PC1"] = train_pca
val_df["PC1"] = val_pca
test_df["PC1"] = test_pca

print(f"Train Verisi Satır Sayısı: {train_df.shape[0]}")
print(f"Validation Verisi Satır Sayısı: {val_df.shape[0]}")
print(f"Test Verisi Satır Sayısı: {test_df.shape[0]}")

from src.models.automata import ProbabilisticAutomata

w_size = config["automata_params"]["window_size"]
a_size = config["automata_params"]["alphabet_size"]

automata = ProbabilisticAutomata(window_size=w_size, alphabet_size=a_size)

train_pc1 = train_df["PC1"].values

sax_dizisi = automata.transform_to_sax(train_pc1)
oruntuler = automata.extract_patterns(sax_dizisi)

automata.fit(oruntuler)

mevcut_durum = oruntuler[0]
gelen_oruntu = "zzzz"

aciklama_json = automata.explain_decision(time_step=5, current_state=mevcut_durum, incoming_pattern=gelen_oruntu, threshold=0.10)

print("\n[SYSTEM DECISION - JSON FORMAT]")
print(json.dumps(aciklama_json, indent=4))