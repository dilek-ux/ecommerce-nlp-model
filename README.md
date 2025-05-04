# ecommerce-nlp-model
TF-IDF ve Word2Vec ile ürün başlıkları üzerinden benzerlik analizi yaptık

**E-Ticaret Projesi: Ürün Açıklama Eşleştirme**

**Problem nedir:**

E-ticaret platformlarında farklı satıcılar, aynı ürünü farklı açıklamalarla listeleyebilmektedir. Bu durum, kullanıcı deneyimini olumsuz etkileyebileceği gibi, ürünlerin doğru şekilde gruplanamamasına da yol açar. Bu projenin amacı, benzer ürün açıklamalarını gruplayarak **ürün kümeleri** oluşturmak ve tekrar eden kayıtları anlamlı bir şekilde bir araya getirmektir.

**Amaç:**
Farklı satıcılara ait açıklamaları analiz ederek, aynı ürünü ifade eden açıklamaları **otomatik olarak gruplayan** bir sistem geliştirmektir.

**Kullanılan Yöntemler ve Adımlar:**

**Veri Kaynağı:**

Bu projede kullanılan ürün açıklamaları verisi, Kaggle platformundaki Shopee - Price Match Guarantee yarışmasından alınmıştır.

Kullanılan dosya: train.csv

İçeriğinde: Ürün başlıkları (title), ürün ID’leri (posting_id), grup kimlikleri (label_group) gibi alanlar yer almaktadır.

Veri, farklı satıcıların aynı ürünü nasıl farklı şekilde adlandırdığını gözlemlemek ve bu açıklamaları gruplayarak eşleştirme yapmak için kullanılmıştır.

Açıklamalar, genellikle marka ve fiyat gibi ek unsurlar içerdiğinden, veri ön işleme gereklidir.

**Kaynak:** Kaggle - Shopee Price Match Guarantee yarışması/ https://www.kaggle.com/competitions/shopee-product-matching/data?select=train.csv
- **Dosya:** 'train.csv' [train (2).csv](https://github.com/user-attachments/files/20028541/train.2.csv)

-**Küçültülmüş dosya** 'train_sample_5000.csv' [train_sample_5000 (2).csv](https://github.com/user-attachments/files/20028542/train_sample_5000.2.csv)

  
**Zipf Yasası Analizi**

Kelimelerin frekans dağılımı incelenerek açıklama yapılarının doğallığı ve bilgi yoğunluğu değerlendirilmiştir.

**Veri Temizleme**

Marka, fiyat, boyut gibi ayırt edici ancak gruplaştırmaya engel olabilecek bilgiler açıklamalardan temizlenmiştir.
  
**Vektörleştirme:**

Her açıklama, içeriğindeki kelimelerin vektörlerinin TF-IDF ağırlıklı ortalaması alınarak temsil edilmiştir.

Açıklamalardan kelime temsilleri oluşturmak için Word2Vec modeli eğitilmiştir.

Böylece her kelimenin açıklamadaki önemi dikkate alınarak daha anlamlı ve ayrım gücü yüksek vektörler elde edilmiştir.


**Benzerlik Ölçümü:**

- **Cosine Similarity** metriği kullanılarak açıklamalar arasındaki benzerlikler hesaplanmıştır.
- Eşik değeri **0.85 üzeri** olan açıklamalar **aynı ürün grubu** olarak belirlenmiştir.

**Kullanılan Teknolojiler:**
-Python         | Projenin ana dili                    
-Pandas / NumPy | Veri işleme ve matematiksel analiz   
-Gensim         | Word2Vec modeli için                 
-Scikit-learn   | TF-IDF ve benzerlik ölçümleri için   
-Matplotlib     | Görselleştirme                       
-Jupyter Notebook | Geliştirme ortamı

**Dosya İçeriği:**

- `ecommerce_nlp_model.ipynb`: Proje adımlarını içeren Jupyter Notebook dosyası
- `README.md`: Bu açıklama dosyası
  
# Nasıl Çalıştırılır?

**Öncelikle gerekli kütüphaneleri yüklüyoruz**

```` 
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import os
````

**Ardından veri setini çekiyoruz**

**Kaynak:** Kaggle - Shopee Price Match Guarantee yarışması/ https://www.kaggle.com/competitions/shopee-product-matching/data?select=train.csv , buradan train.csv verisini çekiyoruz.

````
# Orijinal veri
df = pd.read_csv("train.csv")

# 5 000 satırlık rastgele alt küme
df_sample = df.sample(n=5000, random_state=42)

# Yeni dosyayı kaydet
df_sample.to_csv("train_sample_5000.csv", index=False)

print("Oluşturulan alt küme satır sayısı:", df_sample.shape[0])

# Yeni oluşturduğumuz alt küme dosyasını yükle
df_sample = pd.read_csv("train_sample_5000.csv")


# Veriyi incele
print(df_sample.head())
````



![Ekran görüntüsü 2025-05-04 183050](https://github.com/user-attachments/assets/d81362cc-8b18-46a4-9f61-72c46b475e09)

**ardından** 

````
df_sample = pd.read_csv("train_sample_5000.csv")

# 'title sutunuyla çalışıyoruz
sentences = df_sample['title'].astype(str).tolist()
````
kodunu çalıştırıyoruz.

**Şimdi Zipf Yasası Analizine geçiyoruz**
````
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


# Veriyi yükle (train.csv dosyanız aynı klasörde olmalı)
df_sample = pd.read_csv('train_sample_5000.csv')

# Ham başlıkları tek bir metin haline getir
all_text = ' '.join(df['title'].astype(str))

# Kelimelere ayır (tokenize) ve küçük harfe çevir, sadece harf olanları al
tokens = [word.lower() for word in word_tokenize(all_text) if word.isalpha()]

# Kelime frekanslarını hesapla
word_counts = Counter(tokens)

# Sıklıklara göre sırala
sorted_word_counts = word_counts.most_common()

# Rank (sıra) ve frekans değerlerini çıkar
ranks = np.arange(1, len(sorted_word_counts) + 1)
frequencies = [count for word, count in sorted_word_counts]

# Zipf Yasası grafiği (log-log eksende)
plt.figure(figsize=(10,6))
plt.loglog(ranks, frequencies, marker='.', linestyle='None')
plt.title("Zipf Yasası Analizi (Ham Veri Üzerinden)")
plt.xlabel("Rank (Kelime Sırası)")
plt.ylabel("Frequency (Kelime Frekansı)")
plt.grid(True)
plt.show()
````

![Ekran görüntüsü 2025-05-04 184406](https://github.com/user-attachments/assets/c8ad5406-f315-45af-b191-8b2d65191472)

**Şimdi Temizleme Aşamasına geçiyoruz**

````
import re

# Örnek marka listesi — gerekirse genişletilebilir
brands = ['nike', 'adidas', 'samsung', 'apple', 'xiaomi', 'huawei']

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()  # Lowercase (küçük harfe dönüştür)
    
    # HTML taglerini temizle
    text = re.sub(r'<[^>]+_>', '', text)
    
    # Marka isimlerini kaldır (önce brands listesi tanımlanmış olmalı)
    # Burada sadece listeyi kullanıyoruz ve listede yer alan markaları tamamen temizliyoruz
    text = re.sub(r'(' + '|'.join(map(re.escape, brands)) + r')', '', text)

    # Fiyat ifadelerini kaldır (USD, GBP, EUR, $, £, €)
    text = re.sub(r'\d+(?:\.\d+)?\s?(usd|gbp|eur|\$|£|€)', '', text)
    
    # Ölçü birimlerini kaldır (pound, oz, lb, ml, l, g, mg, inch, in, cm, mm)
    text = re.sub(r'\d+(?:\.\d+)?\s?(pounds?|lbs?|oz|ml|l|g|gr|x|kg|mg|inch|in|cm|mm)', '', text)

    # Kalan tüm sayıları temizle
    text = re.sub(r'\d+', '', text)
    
    # Noktalama işaretlerini kaldır (sadece harf ve rakamlar kalsın)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Fazla boşlukları temizle ve başındaki/sonundaki boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()
 
    return text
````
````
# Lemmatizer başlat
lemmatizer = WordNetLemmatizer()

# POS etiketlerini WordNet formatına dönüştüren fonksiyon
def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ     # Sıfat
    elif tag.startswith('V'): return wordnet.VERB  # Fiil
    elif tag.startswith('R'): return wordnet.ADV   # Zarf
    else: return wordnet.NOUN                      # İsim (varsayılan)

# Lemmatizasyon yapan fonksiyon
def lemmatize_text(text):
    tokens = word_tokenize(text)                  # Kelimelere ayır
    tags = pos_tag(tokens)                        # Her kelimeye tür etiketi ata
    return " ".join([lemmatizer.lemmatize(w, get_pos(t)) for w, t in tags])  # Lemmatize et

# CSV dosyasını oku
df_sample = pd.read_csv("train_sample_5000.csv")

# Temizlenmiş ve kök hâline getirilmiş başlıkları ekle
df_sample["cleaned_title"] = df_sample["title"].astype(str).apply(clean_text)
df_sample["lemmatized_title"] = df_sample["cleaned_title"].apply(lemmatize_text)

# Sonuçları kontrol etmek için ilk 5 satırı yazdır
print(df_sample[["title", "cleaned_title", "lemmatized_title"]].head())
````
![image](https://github.com/user-attachments/assets/a577ee08-a50d-4963-a9e5-90c43f00db2f)

````
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)  # Cümleyi kelimelere ayırdık
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return lemmatized_tokens, stemmed_tokens

# Tüm veriye uyguladık
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in sentences:
    lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)
````
````
# İlk 3 lemmatize edilmiş sonucu gösterdik
print("Lemmatize Edilmiş İlk 3 Cümle:")
for i, tokens in enumerate(tokenized_corpus_lemmatized[:3], start=1):
    print(f"{i}. {tokens}")
````
![Ekran görüntüsü 2025-05-04 185555](https://github.com/user-attachments/assets/c68691f6-2b41-441b-bbd0-057dec07a146)

````
# 1) Önce token listelerini DataFrame’e ekledik
df_sample['tokens_lemmatized'] = tokenized_corpus_lemmatized

# 2) Ardından bu sütunu boşlukla birleştirip CSV’ye yazdırdık
df_sample['tokens_lemmatized'] \
    .apply(lambda lst: ' '.join(lst)) \
    .to_csv('lemmatized_sentences.csv', index=False, header=['lemmatized_text'])

# Stemlenmiş token listelerini DataFrame’e ekledik
df_sample['tokens_stemmed'] = tokenized_corpus_stemmed

# Stemlenmiş cümleleri boşlukla birleştirip CSV’ye yazdırdık
df_sample['tokens_stemmed'] \
    .apply(lambda lst: ' '.join(lst)) \
    .to_csv('stemmed_sentences.csv', index=False, header=['stemmed_text'])
````
````
print("\nTemizlenmiş ve işlenmiş verinin ilk 5 satırı:")
print(df_sample[['title', 'tokens_lemmatized', 'tokens_stemmed']].head(5))
````
![Ekran görüntüsü 2025-05-04 191141](https://github.com/user-attachments/assets/10de7a38-64ef-4be0-956e-f46a77789cdd)

````
from sklearn.feature_extraction.text import TfidfVectorizer
# Ön işlenmiş token listelerini tekrar metne çeviriyoruz
lemmatized_texts = [' '.join(tokens) for tokens in tokenized_corpus_lemmatized]
lemmatized_texts[:3]
````
![image](https://github.com/user-attachments/assets/655a9666-2579-48fe-8d83-f80856268822)

**Şimdi TF-IDF Uygulama Aşamasına Geçiyoruz**
````
# TF-IDF vektörizerı başlatıyoruz
vectorizer = TfidfVectorizer()
# TF-IDF matrisini oluşturuyoruz
# Terim frekansları, belge frekanslarını hesaplar
# TF-IDF vektörlerine dönüştürür
tfidf_matrix = vectorizer.fit_transform(df_sample['lemmatized_title'])

# Kelimeleri alalım
# TF-IDF vektörleştirme işleminde kullanılan tüm kelimelerin eşsiz bir listesini alalım
feature_names = vectorizer.get_feature_names_out()

# TF-IDF matrisini pandas DataFrame'e çevir - görünürlük açısından
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# İlk birkaç satırı gösterelim - ilk 5 cümle
print(tfidf_df.head())

# Sonuçları bir CSV dosyasına kaydedelim
tfidf_df.to_csv("tfidf_lemmatized.csv", index=False)
print("✅ tfidf_lemmatized.csv dosyası kaydedildi.")
````
![Ekran görüntüsü 2025-05-04 191514](https://github.com/user-attachments/assets/41fc554d-515e-4cb7-8db6-8505207028b0)

````
from sklearn.feature_extraction.text import TfidfVectorizer


# df_sample zaten yüklü ve 'lemmatized_title' sütununu içeriyor
# Metinleri listeye dönüştür
texts = df_sample["lemmatized_title"].astype(str).tolist()

# PorterStemmer ile stemming
stemmer = PorterStemmer()
stemmed_texts = []
for text in texts:
    tokens = word_tokenize(text)                       # Cümleyi kelimelere ayır
    stemmed = [stemmer.stem(token) for token in tokens]  # Her kelimenin kökünü al
    stemmed_texts.append(" ".join(stemmed))            # Yeniden cümle hâline getir

# Şimdi TF-IDF işlemini yap
vectorizer_stem = TfidfVectorizer()
tfidf_matrix_stem = vectorizer_stem.fit_transform(stemmed_texts)
feature_names_stem = vectorizer_stem.get_feature_names_out()

# DataFrame'e çevir ve CSV'ye kaydet
tfidf_df_stem = pd.DataFrame(tfidf_matrix_stem.toarray(), columns=feature_names_stem)
tfidf_df_stem.to_csv("tfidf_stemmed.csv", index=False)

# İlk 5 satırı göster
print(tfidf_df_stem.head())
print("✅ tfidf_stemmed.csv dosyası kaydedildi.")
````
````
# df_sample üzerinde 'lemmatized_title' sütununu kullanıyoruz
tfidf = TfidfVectorizer()
df_tfidf = pd.DataFrame(
    tfidf.fit_transform(df_sample["lemmatized_title"].astype(str)).toarray(),
    columns=tfidf.get_feature_names_out()
)

print("İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(df_tfidf.iloc[0].sort_values(ascending=False).head(5))
````
![Ekran görüntüsü 2025-05-04 191703](https://github.com/user-attachments/assets/811b6a9e-3faa-48de-808f-2111a79852b0)

````
# PorterStemmer nesnesi
stemmer = PorterStemmer()

# Her başlık için stemming uygula
def stem_text(text):
    tokens = word_tokenize(str(text))
    stems  = [stemmer.stem(t) for t in tokens]
    return " ".join(stems)

# Yeni sütunu ekle
df_sample["stemmed_title"] = df_sample["lemmatized_title"].apply(stem_text)


tfidf = TfidfVectorizer()
# Artık sütun var, doğrudan kullan
tfidf_matrix = tfidf.fit_transform(df_sample["stemmed_title"].astype(str))
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

print("İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(df_tfidf.iloc[0].sort_values(ascending=False).head(5))

````
![Ekran görüntüsü 2025-05-04 191803](https://github.com/user-attachments/assets/56b2bce3-763c-4595-84ab-2fdf02d13f18)

**Cosine Similarity benzerliğini hesaplıyoruz**

````
from sklearn.metrics.pairwise import cosine_similarity

vectorizer_stem = TfidfVectorizer()
tfidf_matrix_stem = vectorizer_stem.fit_transform(stemmed_texts)

# 2. Cosine benzerlik matrisi
cosine_sim = cosine_similarity(tfidf_matrix_stem)

# 3. Gruplama işlemi
groups = []
visited = set()
threshold = 0.85

for i in range(len(stemmed_texts)):
    if i in visited:
        continue
    group = [i]
    visited.add(i)
    for j in range(i + 1, len(stemmed_texts)):
        if cosine_sim[i][j] > threshold:
            group.append(j)
            visited.add(j)
    groups.append(group)

# 4. group_id'leri orijinal veriye ekle
group_ids = [None] * len(stemmed_texts)
for group_id, group in enumerate(groups):
    for idx in group:
        group_ids[idx] = group_id

df_sample = df_sample.iloc[:len(stemmed_texts)].copy()  # dataframe
df_sample["group_id"] = group_ids

# 5. Kontrol
print(df_sample[["cleaned_title", "group_id"]].head(20))
````
![image](https://github.com/user-attachments/assets/acd55923-1adf-46c5-b88b-d68edc1fc48e)

````
# 1) TF-IDF matrisini df_sample üzerinden yeniden oluşturuyoruz
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_sample["lemmatized_title"].astype(str))
feature_names = vectorizer.get_feature_names_out()

# 2) "creamer" kelimesinin TF-IDF vektörünü al
creamer_index = list(feature_names).index("creamer")          # 'creamer' indeksini bul
creamer_vector = tfidf_matrix[:, creamer_index].toarray()    # o terimin vektörü

# 3) Tüm terimlerin vektörlerini kullanarak sütunlar arası benzerlik hesapladık
#    (tfidf_matrix.T ile sütun bazlı vektörler elde ediyoruz)
similarities = cosine_similarity(tfidf_matrix.T[creamer_index], tfidf_matrix.T).flatten()

# 4) En yüksek 5 benzer terimi al (kendisi de geleceği için 6 alıp ilkini atacağız)
top_idxs = similarities.argsort()[-6:][::-1]
top_idxs = [i for i in top_idxs if i != creamer_index][:5]

# 5) Sonuçları yazdırdık
print("‘creamer’ kelimesine en çok benzeyen 5 kelime:")
for idx in top_idxs:
    print(f"{feature_names[idx]}: {similarities[idx]:.4f}")
````
![Ekran görüntüsü 2025-05-04 191959](https://github.com/user-attachments/assets/c860f093-be16-4130-838d-840bfc0b7cf3)

**Word2Vec modeline geçiyoruz**

````
df = pd.read_csv("train_sample_5000.csv")
texts = df_sample["title"].dropna().astype(str).tolist()  # eksik değer kontrolü

# Tokenize edilmiş cümle listesi
tokenized_texts = [word_tokenize(text.lower()) for text in texts]
````
![Ekran görüntüsü 2025-05-04 192253](https://github.com/user-attachments/assets/48f3d09e-8490-4ab7-8eeb-1e39d8fc3918)

````
# 1) df_sample’dan tokenized corpus’ları oluştur
tokenized_corpus_lemmatized = df_sample["lemmatized_title"] \
    .dropna().astype(str) \
    .apply(lambda t: word_tokenize(t.lower())) \
    .tolist()

tokenized_corpus_stemmed = df_sample["stemmed_title"] \
    .dropna().astype(str) \
    .apply(lambda t: word_tokenize(t.lower())) \
    .tolist()

# 2) Parametre setleri
parameters = [
    {'model_type': 'cbow',     'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow',     'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow',     'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow',     'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# 3) Model eğitme ve kaydetme fonksiyonu
def train_and_save_model(corpus, params, prefix):
    sg_flag = 1 if params['model_type']=='skipgram' else 0
    model = Word2Vec(
        sentences=corpus,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=1,
        sg=sg_flag,
        workers=4,
        epochs=10
    )
    fname = f"{prefix}_{params['model_type']}_w{params['window']}_d{params['vector_size']}.model"
    model.save(fname)
    print(f"→ Saved {fname}")

# 4) Eğit ve kaydet
for p in parameters:
    train_and_save_model(tokenized_corpus_lemmatized, p, "lemmatized_model")

for p in parameters:
    train_and_save_model(tokenized_corpus_stemmed,  p, "stemmed_model")
````
![Ekran görüntüsü u2025-05-04 192413](https://github.com/user-attachments/assets/bbed076a-6ab3-4fe2-80ad-78b3fbb097a0)

````
# Modelleri yükledik
model_1 = Word2Vec.load("lemmatized_model_cbow_window2_dim100.model")
model_2 = Word2Vec.load("stemmed_model_skipgram_window4_dim100.model")
model_3 = Word2Vec.load("lemmatized_model_skipgram_window2_dim300.model")

# Benzer kelimeleri yazdıran fonksiyon
def print_similar_words(model, model_name, keyword='creamer'):
    try:
        similarity = model.wv.most_similar(keyword, topn=3)
        print(f"\n{model_name} Modeli - '{keyword}' ile En Benzer 3 Kelime:")
        for word, score in similarity:
            print(f"Kelime: {word}, Benzerlik Skoru: {score:.4f}")
    except KeyError:
        print(f"\n{model_name} Modeli: '{keyword}' kelimesi modelin kelime dağarcığında bulunamadı.")

# Her model için fonksiyonu çağırdık
print_similar_words(model_1, "Lemmatized CBOW Window 2 Dim 100")
print_similar_words(model_2, "Stemmed Skipgram Window 4 Dim 100")
print_similar_words(model_3, "Lemmatized Skipgram Window 2 Dim 300")
````
![Ekran görüntüsü 2025-05-04 192512](https://github.com/user-attachments/assets/c40e98c4-8646-4864-8bfa-28a1bf98c987)

# sizlerde buradaki adımları takip ederek kendi kodlarınızı çalıştırabilirsiniz, Başarılar


















  


