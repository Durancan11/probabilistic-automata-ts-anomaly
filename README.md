# Olasılıksal Otomata ile Zaman Serisi Anomali Tespiti ve Karşılaştırmalı Analiz

Bu proje, zaman serisi verilerinde anomali tespiti problemini çözmek için geliştirilmiş iki farklı yapay zeka paradigmasını (**White-Box** ve **Black-Box**) karşılaştırmalı olarak analiz etmeyi amaçlamaktadır. Proje kapsamında fiziksel arızaları temsil eden **SKAB** ve siber-fiziksel saldırıları temsil eden **BATADAL** veri setleri kullanılmıştır.

## 📌 Projenin Amacı ve Felsefesi
Bu çalışmanın temel amacı sadece en yüksek doğruluk (accuracy) veya F1 skoruna ulaşmak değil; farklı mimarilerin spesifik veri karakteristiklerine nasıl tepki verdiğini incelemektir. 
* **Black-Box Modeller (LSTM & GRU):** Yüksek hesaplama gücü ve karmaşık örüntü yakalama kapasitesine sahiptir, ancak kararlarının arkasındaki mantığı açıklayamazlar.
* **White-Box Model (Probabilistic Automata):** Kararlarını geçiş olasılıklarına (Transition Probabilities) dayandırır ve bir anomali tespit ettiğinde bunun "neden" anomali olduğunu matematiksel ve mantıksal olarak açıklayabilir.

## ⚙️ Metodoloji ve Geliştirme Süreci

### 1. Veri Ön İşleme ve Boyut İndirgeme (PCA)
Otomata modelinin karmaşıklığını yönetilebilir seviyede tutmak için çok değişkenli sensör verileri **PCA (Principal Component Analysis)** kullanılarak tek boyuta (PC1) indirgenmiştir. Derin öğrenme modelleri ise verinin orijinal çok boyutlu halini kullanmıştır.

### 2. Olasılıksal Otomata (SAX & PAA)
Zaman serisi verileri PAA (Piecewise Aggregate Approximation) ve SAX (Symbolic Aggregate approXimation) kullanılarak sembolik harflere dönüştürülmüştür. (Örn: `window_size=4`, `alphabet_size=3`). Model, eğitim verisindeki "normal" geçiş olasılıklarını öğrenerek bir durum makinesi (State Machine) inşa etmiştir.

### 3. Dinamik Eşik (Dynamic Thresholding & Percentile)
Sabit bir eşik değerinin her veri setine uymayacağı saptanmıştır. Özellikle BATADAL gibi 81 farklı durumun (state) yaşandığı karmaşık ağlarda olasılıklar doğal olarak küçülmektedir. Bu sorunu çözmek için **İstatistiksel Yüzdelik Dilim (Percentile)** yöntemi kullanılmıştır. Eğitimdeki geçiş olasılıklarının en nadir %10'luk dilimi dinamik olarak hesaplanmış ve anomali sınırı (threshold) olarak belirlenmiştir.

### 4. Derin Öğrenme Eğitimi ve Early Stopping
LSTM ve GRU modelleri eğitilirken ezberlemenin (overfitting) önüne geçmek amacıyla **Early Stopping** (patience=5) mekanizması entegre edilmiştir. Sınıf dengesizliğini çözmek için eğitimde `class_weight` parametresi kullanılmıştır.

---

## 📊 Deneysel Analizler ve Çarpıcı Bulgular

Proje testleri sonucunda veri setlerinin doğasından kaynaklanan çok net davranış farklılıkları gözlemlenmiştir:

### SKAB Veri Seti (Fiziksel Arızalar)
* **Karakteristik:** Vanalardaki fiziksel arızalar, sinyallerde ani ve çok belirgin kırılmalara yol açar.
* **Bulgular:** Sinyaller belirgin olduğu için LSTM ve GRU modelleri **~0.83 F1 Skoru** ile çok başarılı sonuçlar vermiştir. Otomata modeli, PCA ile verinin tek boyuta inmesinin dezavantajını yaşasa da dinamik threshold sayesinde anomalileri yakalayabilmiştir.

### BATADAL Veri Seti (Siber-Fiziksel Saldırılar)
* **Karakteristik:** Bir su şebekesine yapılan siber saldırılar, algoritmaları kandırmak için **"normal veriyi taklit edecek şekilde"** (sinsi ve gizli) tasarlanmıştır.
* **Bulgular:** LSTM ve GRU gibi Black-Box modeller bu sinsi manipülasyonlar karşısında adeta körleşmiş ve risk almamak için her duruma "Normal (0)" deme eğilimi göstermiştir. Yüksek Accuracy (%85) vermelerine rağmen F1 skorları **~0.03** seviyelerinde çakılmıştır.
* **Otomata'nın Başarısı:** Derin öğrenme modellerinin sınıfta kaldığı bu veri setinde, Olasılıksal Otomata **0.05 F1 Skoru** üreterek rakiplerini geride bırakmıştır. Confusion Matrix incelendiğinde, Automata'nın çok daha az "Yanlış Alarm (False Positive)" ürettiği ve temkinli yapısının siber saldırı tespitinde Black-Box modellere kıyasla daha güvenilir bir referans oluşturduğu kanıtlanmıştır.

---

## 📈 Görselleştirme ve Hata Analizi (Visuals)

Proje kapsamında modellerin BATADAL veri seti üzerindeki karakterleri ve zafiyetleri görselleştirilmiştir.

### 1. Confusion Matrix (Karmaşıklık Matrisi) Karşılaştırması
Siber saldırı (gizli manipülasyon) senaryolarında modellerin verdiği tepkiler ve ürettikleri Yanlış Alarm (False Positive) oranları karşılaştırılmıştır.

**Olasılıksal Otomata (White-Box):**
*Derin öğrenme modellerine kıyasla çok daha az "Yanlış Alarm" (Sadece 35 FP) üreterek temkinli yapısını kanıtlamıştır. Black-box modellere göre çok daha güvenilir bir referans noktası sunar.*
![Automata Confusion Matrix](Automata.png)

**LSTM Modeli (Black-Box):**
*Dengesiz veri setinde anomali yakalamaya çalışırken kontrolü kaybetmiş ve yüksek oranda yanlış alarm (50 FP) üretmiştir.*
![LSTM Confusion Matrix](LSTM.png)

**GRU Modeli (Black-Box):**
*Benzer şekilde yüksek yanlış alarm (73 FP) oranıyla sinsi siber saldırı verilerinde en çok zorlanan model olmuştur.*
![GRU Confusion Matrix](GRU.png)

---

### 2. Precision-Recall (PR) Eğrileri
Derin öğrenme modellerinin (LSTM ve GRU) eşik değerleri değiştikçe gösterdikleri performansı analiz etmek için PR eğrileri çizdirilmiştir. 

Grafiklerdeki karakteristik "L" şekilli ani çöküş; modelin Recall (yakalama oranı) değerini azıcık bile artırmaya çalıştığında Precision (isabet oranı) değerinin anında sıfıra indiğini kanıtlamaktadır. Bu durum, Black-Box modellerin dengesiz zaman serilerindeki (imbalanced time-series) ezberci zafiyetini görsel olarak ispatlar.

**LSTM PR Eğrisi:**
![LSTM PR Curve](LSTM%20Grafik.png)

**GRU PR Eğrisi:**
![GRU PR Curve](GRU%20Grafik.png)

## 🔍 Olasılıksal Açıklanabilirlik Modülü (Explainability JSON)

Automata modelinin en büyük gücü olan "açıklanabilirlik" özelliği, sisteme entegre edilen bir JSON modülü ile kanıtlanmıştır. Bir anomali (veya normal durum) tespit edildiğinde sistem arka planda şu çıktıyı üretir:

```json
{
    "time_step": 6,
    "state": "caaa",
    "pattern": "aaaa",
    "status": "seen",
    "mapped_to": "aaaa",
    "probability": 0.885,
    "decision": "normal"
}

