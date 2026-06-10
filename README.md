# 🧠 Sentiment Analysis Studio for Turkish NLP

<p align="center">
  <img src="assets/ChatBot%20screenshot.jpg" alt="Sentiment Analysis Studio Dashboard" width="800">
</p>

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

🌍 *[Türkçe açıklama için aşağı kaydırın / Scroll down for Turkish](#türkçe-açıklama)*

---

## 📌 Project Overview
Sentiment Analysis Studio is an end-to-end Machine Learning pipeline designed specifically for the morphological complexities of the Turkish language. Rather than relying on computationally expensive Large Language Models (LLMs), this project utilizes a highly optimized, deterministic ML pipeline with **TF-IDF N-gram vectorization** and **Logistic Regression**, achieving millisecond-level inference times.

## 🚀 Key Engineering Features
* **Agglutinative Language Handling:** Custom text preprocessing that handles Turkish capitalization errors and noise without removing critical stop-words (e.g., "değil", "ama") that shift sentiment context.
* **Strict Data Leakage Protection:** The pipeline enforces strict separation between training and testing. Vocabulary and IDF weights are built exclusively on the Train Set (`fit_transform`), simulating a true production environment for the Test Set (`transform`).
* **Live Active Learning Loop:** Features an SQLite-integrated feedback mechanism. Misclassified texts can be corrected via the UI, triggering an **in-memory retraining** of the entire pipeline (TF-IDF + Model) in seconds without system downtime.
* **Decoupled Architecture:** Text processing and classification layers are completely independent, allowing for future algorithmic swapping without breaking the system.

## 📊 Model Performance
The model was evaluated using a Stratified Split dataset (80% Train, 20% Test) to ensure balanced class distribution.
* **Overall Accuracy:** `90.91%`
* **Positive (F1-Score):** `0.947`
* **Neutral (F1-Score):** `0.909`
* **Negative (F1-Score):** `0.880`

---

<a id="türkçe-açıklama"></a>
# 🧠 Sentiment Analysis Studio (Türkçe NLP)

## 📌 Proje Özeti
Sentiment Analysis Studio, Türkçe'nin eklemeli yapısına ve sosyal medya verilerindeki gürültülere özel olarak tasarlanmış uçtan uca bir makine öğrenmesi boru hattıdır (pipeline). Yüksek hesaplama maliyeti gerektiren büyük dil modelleri (LLM) yerine, **TF-IDF N-gram vektör uzayı** ve **Lojistik Regresyon** tabanlı, milisaniyeler seviyesinde çalışan optimize bir sistem kullanılmıştır.

## 🚀 Öne Çıkan Mühendislik Çözümleri
* **Gelişmiş Ön İşleme:** Noktalama işaretleri ve gürültüler Regex ile temizlenirken, geleneksel NLP yaklaşımlarının aksine duygu yönünü değiştiren "değil", "ama" gibi bağlaçlar bilinçli olarak veri setinde tutulmuştur.
* **Veri Sızıntısı (Data Leakage) Koruması:** Modelin test setini ezberlemesini önlemek için katı mimari kurallar uygulanmıştır. Kelime haznesi sadece eğitim seti üzerinden (`fit_transform`) oluşturulmuş, test verisi canlı ortam verisi gibi simüle edilmiştir.
* **Canlı Aktif Öğrenme (Active Learning):** Sistem arayüzü üzerinden yanlış tahmin edilen veriler düzeltilebilir. Bu düzeltmeler SQLite veritabanına işlenir ve sistem arka planda tüm boru hattını **(In-Memory Retraining)** saniyeler içinde yeniden eğiterek sürekli evrilir.
* **Bağımsız (Decoupled) Mimari:** Modüller birbirinden bağımsız tasarlanmıştır, bu sayede veri işleme veya algoritma katmanı diğer bileşenleri bozmadan değiştirilebilir.

## 📊 Model Performansı
Metrikler, dengesiz dağılımları önlemek adına Stratified Split (Tabakalı Bölme) yöntemiyle %80 Eğitim ve %20 Test olarak ayrılan veri setinde elde edilmiştir.
* **Genel Doğruluk (Accuracy):** `%90.91`
* **Pozitif (F1-Skoru):** `0.947`
* **Nötr (F1-Skoru):** `0.909`
* **Negatif (F1-Skoru):** `0.880`

## 💻 Kurulum ve Kullanım (Installation & Usage)
1. Repoyu bilgisayarınıza klonlayın:
```bash
   git clone [https://github.com/KULLANICI_ADIN/ChatBot-for-Sentiment-Analysis.git](https://github.com/KULLANICI_ADIN/ChatBot-for-Sentiment-Analysis.git)
