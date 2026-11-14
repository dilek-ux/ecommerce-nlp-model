**Ecommerce-nlp-model**

TF-IDF ve Word2Vec ile ürün başlıkları üzerinden benzerlik analizi yaptık

**E-Ticaret Projesi: Ürün Açıklama Eşleştirme**

**Problem nedir:**

E-ticaret platformlarında farklı satıcılar, aynı ürünü farklı açıklamalarla listeleyebilmektedir. Bu durum, kullanıcı deneyimini olumsuz etkileyebileceği gibi, ürünlerin doğru şekilde gruplanamamasına da yol açar. Bu projenin amacı, benzer ürün açıklamalarını gruplayarak **ürün kümeleri** oluşturmak ve tekrar eden kayıtları anlamlı bir şekilde bir araya getirmektir.

**Amaç:**
Farklı satıcılara ait açıklamaları analiz ederek, aynı ürünü ifade eden açıklamaları **otomatik olarak gruplayan** bir sistem geliştirmektir.

**Kullanılan Yöntemler ve Adımlar:**

## Veri Kaynağı

Bu projede kullanılan ürün verileri, PriceRunner adlı ürün karşılaştırma platformundan elde edilen **PriceRunner Product Matching (Product Classification and Clustering)** veri setinden alınmıştır.

**Kullanılan dosya:** `pricerunner_aggregate.csv` (PriceRunner Birleştirilmiş Ürün Verileri)

**Veri setinin içeriği:**

- `Product ID`: Ürün kimliği.
- `Product Title`: Satıcılar tarafından oluşturulan ürün başlıkları.
- `Vendor ID`: Ürünü listeleyen satıcının kimliği.
- `Cluster ID`: Aynı ürüne ait kayıtları gruplandıran kimlik.
- `Cluster Label`: Ürünün platformdaki standart adı.
- `Category ID`: Ürün kategorisinin kimliği.
- `Category Label`: Ürün kategorisinin adı.

Veri seti, 306 farklı satıcı tarafından sağlanan 35.311 ürün kaydından ve 10 farklı ürün kategorisinden oluşmaktadır. Ürün başlıkları İngilizcedir.

Bu veri seti, farklı satıcıların aynı ürünü farklı başlıklarla nasıl tanımladığını incelemek, metin benzerliği hesaplamak ve aynı ürüne ait kayıtları eşleştirmek amacıyla kullanılacaktır.

Proje kapsamında ürün başlıkları üzerinde metin ön işleme işlemleri gerçekleştirilecek; Tokenization, Stopword Removal, Lemmatization ve Stemming yöntemleri uygulanacaktır. Ardından TF-IDF, Word2Vec ve Cosine Similarity yöntemleriyle ürün başlıkları arasındaki benzerlikler incelenecektir.

**Kaynak:** PriceRunner Product Matching Dataset, Hugging Face / UCI Machine Learning Repository.

**Veri seti bağlantısı:** https://huggingface.co/datasets/lakritidis/product-matching

**CSV dosyası:** `pricerunner_aggregate.csv`
  
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

**Kullanılan Kütüphaneler:**
-Python          :Projenin ana dili                    
-Pandas / NumPy :Veri işleme ve matematiksel analiz   
-Gensim          : Word2Vec modeli için                 
-Scikit-learn   :TF-IDF ve benzerlik ölçümleri için   
-Matplotlib     :Görselleştirme                       
-Jupyter Notebook : Geliştirme ortamı

**Dosya İçeriği:**

- `ecommerce_nlp_model(6).ipynb`: Proje adımlarını içeren Jupyter Notebook dosyası
- `README.md`: Bu açıklama dosyası
  
# Nasıl Çalıştırılır?

**jupyter notebook üzerinden çalışma yapılmıştır**.

**Öncelikle gerekli kütüphaneleri yüklüyoruz**

```` 
# Veri analizi
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Doğal Dil İşleme (NLP)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Metin benzerliği ve özellik çıkarımı
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Word2Vec
from gensim.models import Word2Vec

# Gerekli NLTK kaynakları
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
````

**Ardından veri setini çekiyoruz**

**Kaynak:** PriceRunner Product Matching Dataset, Hugging Face / UCI Machine Learning Repository.
, buradan `pricerunner_aggregate.csv` verisini çekiyoruz.

````
# PriceRunner veri setini yükle
df = pd.read_csv("pricerunner_aggregate.csv")

# Veri setinin boyutunu incele
print("Veri setinin satır ve sütun sayısı:", df.shape)

# İlk 5 kaydı görüntüle
print("\nİlk 5 kayıt:")
print(df.head())

# Sütun bilgilerini incele
print("\nSütun bilgileri:")
print(df.info())
````


<img width="1478" height="747" alt="Ekran görüntüsü 2026-11-14 175957" src="https://github.com/user-attachments/assets/4b63a12e-27c5-4800-9d04-95224a7795f3" />


**ardından kodunu çalıştırıyoruz.** 

**Şimdi Zipf Yasası Analizine geçiyoruz**
````
# Gerekli kütüphaneler
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Ürün başlıklarını tek bir metin haline getir
all_text = ' '.join(df['Product Title'].astype(str))

# Kelimelere ayır ve küçük harfe çevir
tokens = [
    word.lower()
    for word in word_tokenize(all_text)
    if word.isalpha()
]

# Kelime frekanslarını hesapla
word_counts = Counter(tokens)

# Kelimeleri frekanslarına göre sırala
sorted_word_counts = word_counts.most_common()

# Rank ve frekans değerlerini oluştur
ranks = np.arange(1, len(sorted_word_counts) + 1)
frequencies = [count for word, count in sorted_word_counts]

# Zipf Yasası grafiği
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker='.', linestyle='None')

plt.title("Zipf Yasası Analizi (Ham Ürün Başlıkları)")
plt.xlabel("Rank (Kelime Sırası)")
plt.ylabel("Frequency (Kelime Frekansı)")
plt.grid(True)
plt.show()
````

<img width="1506" height="817" alt="Ekran görüntüsü 2026-11-14 180321" src="https://github.com/user-attachments/assets/7b3db65b-58b2-402e-817a-f74bf88c2e1e" />


**Şimdi Temizleme Aşamasına geçiyoruz**

````
import re

def clean_text(text):
    if pd.isna(text):
        return ""

    # Küçük harfe dönüştür
    text = text.lower()

    # HTML etiketlerini temizle
    text = re.sub(r'<[^>]+>', ' ', text)

    # Para birimi ifadelerini temizle
    text = re.sub(r'[$£€]', ' ', text)
    text = re.sub(r'\b(usd|gbp|eur)\b', ' ', text)

    # Sayı ile birim arasındaki boşluğu kaldır
    text = re.sub(r'(\d+)\s+(gb|tb|mb|kg|mg|ml|cm|mm)', r'\1\2', text)

    # Noktalama işaretlerini temizle
    # Harf, rakam ve boşlukları koru
    text = re.sub(r'[^\w\s.]', ' ', text) 

    # Sayıların arasındaki ondalık noktaları koru
    text = re.sub(r'(?<=\d)\.(?=\d)', 'DECIMALPOINT', text)
    text = re.sub(r'\.', ' ', text)
    text = text.replace('DECIMALPOINT', '.')

    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Temizlenmiş ürün başlıklarını yeni sütuna kaydet
df['Cleaned Title'] = df['Product Title'].apply(clean_text)

# Örnek sonuçları incele
print(df[['Product Title', 'Cleaned Title']].head(10))
````
<img width="1492" height="750" alt="Ekran görüntüsü 2026-11-14 180901" src="https://github.com/user-attachments/assets/9157093b-ca23-4593-bfc0-c748cc66f763" />

````
# Gerekli NLTK bileşenleri
from nltk.corpus import wordnet
from nltk import pos_tag

# Lemmatizer başlat
lemmatizer = WordNetLemmatizer()

# POS etiketlerini WordNet formatına dönüştür
def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatization fonksiyonu
def lemmatize_text(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)

    lemmatized_words = [
        lemmatizer.lemmatize(word, get_pos(tag))
        for word, tag in tags
    ]

    return " ".join(lemmatized_words)

# Temizlenmiş başlıklara Lemmatization uygula
df['Lemmatized Title'] = df['Cleaned Title'].apply(lemmatize_text)

# Sonuçları karşılaştır
print(df[['Product Title', 'Cleaned Title', 'Lemmatized Title']].head(10))

````
<img width="1520" height="792" alt="Ekran görüntüsü 2026-11-14 181316" src="https://github.com/user-attachments/assets/ad43d4ed-5ec6-4f8c-9f2c-b6444829794a" />


````
# Gerekli araçları başlat
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# İngilizce stopword listesi
stop_words = set(stopwords.words('english'))

# Ürün eşleştirmede anlamlı olabilecek kelimeleri koru
stop_words -= {'no', 'not', 'nor', 'without'}

# Metin ön işleme fonksiyonu
def preprocess_sentence(sentence):

    # Tokenization
    tokens = word_tokenize(sentence)

    # Stopword temizleme
    filtered_tokens = [
        token.lower()
        for token in tokens
        if token.isalnum() and token.lower() not in stop_words
    ]

    # POS etiketleme
    alphabetic_tokens = [
        token for token in filtered_tokens if token.isalpha()
    ]

    pos_tags = pos_tag(alphabetic_tokens)
    pos_dict = {
        word: tag for word, tag in pos_tags
    }

    # Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(
            token,
            get_pos(pos_dict[token])
        ) if token.isalpha() else token
        for token in filtered_tokens
    ]

    # Stemming
    stemmed_tokens = [
        stemmer.stem(token) if token.isalpha() else token
        for token in filtered_tokens
    ]

    return filtered_tokens, lemmatized_tokens, stemmed_tokens


# PriceRunner verisine uygula
tokenized_corpus = []
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in df['Cleaned Title'].fillna(''):

    tokens, lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)

    tokenized_corpus.append(tokens)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)


# İlk 3 sonucu karşılaştır
for i in range(3):

    print(f"\nÜrün {i+1}")
    print("Orijinal:", df['Product Title'].iloc[i])
    print("Tokenized:", tokenized_corpus[i])
    print("Lemmatized:", tokenized_corpus_lemmatized[i])
    print("Stemmed:", tokenized_corpus_stemmed[i])
````
<img width="1497" height="702" alt="Ekran görüntüsü 2026-11-14 181820" src="https://github.com/user-attachments/assets/397a9eb2-9f7e-49c8-86d5-e1f0ab1a0adf" />


````
# Lemmatization sonuçlarını DataFrame'e ekle
df['Tokens Lemmatized'] = tokenized_corpus_lemmatized

# Lemmatization sonuçlarını CSV dosyasına kaydet
df['Tokens Lemmatized'].apply(
    lambda tokens: ' '.join(tokens)
).to_csv(
    'lemmatized_sentences.csv',
    index=False,
    header=['lemmatized_text']
)


# Stemming sonuçlarını DataFrame'e ekle
df['Tokens Stemmed'] = tokenized_corpus_stemmed

# Stemming sonuçlarını CSV dosyasına kaydet
df['Tokens Stemmed'].apply(
    lambda tokens: ' '.join(tokens)
).to_csv(
    'stemmed_sentences.csv',
    index=False,
    header=['stemmed_text']
)


# Sonuçları kontrol et
print("Temizlenmiş ve işlenmiş verinin ilk 5 satırı:")

print(df[[
    'Product Title',
    'Tokens Lemmatized',
    'Tokens Stemmed'
]].head(5))
````
<img width="1490" height="777" alt="image" src="https://github.com/user-attachments/assets/9eda3c4a-e7bf-4cb3-87fe-d24304ae40b2" />


[lemmatized_sentences (1).csv](https://github.com/user-attachments/files/34854683/lemmatized_sentences.1.csv)

[stemmed_sentences (1).csv](https://github.com/user-attachments/files/34854691/stemmed_sentences.1.csv)

````
# Lemmatize edilmiş kelimeleri birleştir
lemmatized_tokens = [
    token
    for sublist in tokenized_corpus_lemmatized
    for token in sublist
]

# Kelime frekanslarını hesapla
lemmatized_counts = Counter(lemmatized_tokens)

# Frekanslara göre sırala
sorted_lemmatized = lemmatized_counts.most_common()

# Rank ve frekans değerlerini oluştur
ranks_lem = np.arange(1, len(sorted_lemmatized) + 1)
frequencies_lem = [
    count for word, count in sorted_lemmatized
]

# Zipf grafiği
plt.figure(figsize=(10, 6))
plt.loglog(
    ranks_lem,
    frequencies_lem,
    marker='.',
    linestyle='None',
    color='green'
)

plt.title("Zipf Yasası Analizi (Lemmatized Veriler)")
plt.xlabel("Rank (Kelime Sırası)")
plt.ylabel("Frequency (Kelime Frekansı)")
plt.grid(True)
plt.show()
````
<img width="1478" height="786" alt="Ekran görüntüsü 2026-11-14 182735" src="https://github.com/user-attachments/assets/bda3718d-09ea-4745-a720-89b4e10093db" />

````
# Stemlenmiş kelimeleri birleştir
stemmed_tokens = [
    token
    for sublist in tokenized_corpus_stemmed
    for token in sublist
]

# Kelime frekanslarını hesapla
stemmed_counts = Counter(stemmed_tokens)

# Frekanslara göre sırala
sorted_stemmed = stemmed_counts.most_common()

# Rank ve frekans değerlerini oluştur
ranks_stem = np.arange(1, len(sorted_stemmed) + 1)
frequencies_stem = [
    count for word, count in sorted_stemmed
]

# Zipf grafiği
plt.figure(figsize=(10, 6))
plt.loglog(
    ranks_stem,
    frequencies_stem,
    marker='.',
    linestyle='None',
    color='orange'
)

plt.title("Zipf Yasası Analizi (Stemmed Veriler)")
plt.xlabel("Rank (Kelime Sırası)")
plt.ylabel("Frequency (Kelime Frekansı)")
plt.grid(True)
plt.show()
````
<img width="1512" height="798" alt="Ekran görüntüsü 2026-11-14 182910" src="https://github.com/user-attachments/assets/352e4c95-a2a7-4deb-bbf4-e15029518790" />

````
# Lemmatize edilmiş token listelerini metne dönüştür
lemmatized_texts = [
    ' '.join(tokens)
    for tokens in tokenized_corpus_lemmatized
]

# İlk 3 metni görüntüle
print("İlk 3 Lemmatized Metin:\n")

for text in lemmatized_texts[:3]:
    print(text)
    print("-" * 50)
````
<img width="1472" height="556" alt="Ekran görüntüsü 2026-11-14 183111" src="https://github.com/user-attachments/assets/6a1e5154-ed22-4800-9205-baafce6a1a41" />


**Şimdi TF-IDF Uygulama Aşamasına Geçiyoruz**
````
# TF-IDF vektörizerını başlat
vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

# Lemmatized metinlerden TF-IDF matrisi oluştur
tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)

# Kelimeleri al
feature_names = vectorizer.get_feature_names_out()

# Matris boyutunu kontrol et
print("TF-IDF matris boyutu:", tfidf_matrix.shape)
print("Toplam kelime sayısı:", len(feature_names))

# İlk 5 satırı görüntüle
tfidf_preview = pd.DataFrame(
    tfidf_matrix[:5].toarray(),
    columns=feature_names
)

print("\nİlk 5 ürünün TF-IDF değerleri:")
print(tfidf_preview)

# Seyrek matrisi uzun formatlı CSV olarak kaydet
coo_matrix = tfidf_matrix.tocoo()

tfidf_lemmatized_sparse = pd.DataFrame({
    'Product ID': df['Product ID'].to_numpy()[coo_matrix.row],
    'Word': feature_names[coo_matrix.col],
    'TF-IDF Score': coo_matrix.data
})

tfidf_lemmatized_sparse.to_csv(
    "tfidf_lemmatized.csv",
    index=False
)

print("\ntfidf_lemmatized.csv dosyası kaydedildi.")
````
<img width="1495" height="807" alt="Ekran görüntüsü 2026-11-14 194313" src="https://github.com/user-attachments/assets/5eda5857-fca8-4bd4-a543-3fb247db41bc" />


[tfidf_lemmatized (2).csv](https://github.com/user-attachments/files/34856431/tfidf_lemmatized.2.csv)



````
# Stemmed token listelerini metne dönüştür
stemmed_texts = [
    ' '.join(tokens)
    for tokens in tokenized_corpus_stemmed
]

# TF-IDF vektörizerını başlat
vectorizer_stem = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

# Stemmed metinlerden TF-IDF matrisi oluştur
tfidf_matrix_stem = vectorizer_stem.fit_transform(stemmed_texts)

# Kelimeleri al
feature_names_stem = vectorizer_stem.get_feature_names_out()

# Matris boyutunu kontrol et
print("Stemmed TF-IDF matris boyutu:", tfidf_matrix_stem.shape)
print("Toplam kelime sayısı:", len(feature_names_stem))

# İlk 5 ürünün TF-IDF değerlerini görüntüle
tfidf_preview_stem = pd.DataFrame(
    tfidf_matrix_stem[:5].toarray(),
    columns=feature_names_stem
)

print("\nİlk 5 ürünün Stemmed TF-IDF değerleri:")
print(tfidf_preview_stem)

# Seyrek matrisi uzun formatlı CSV olarak kaydet
coo_matrix = tfidf_matrix_stem.tocoo()

tfidf_stemmed_sparse = pd.DataFrame({
    'Product ID': df['Product ID'].to_numpy()[coo_matrix.row],
    'Word': feature_names_stem[coo_matrix.col],
    'TF-IDF Score': coo_matrix.data
})

tfidf_stemmed_sparse.to_csv(
    "tfidf_stemmed.csv",
    index=False
)

print("\ntfidf_stemmed.csv dosyası kaydedildi.")
````
<img width="1498" height="532" alt="Ekran görüntüsü 2026-11-14 194522" src="https://github.com/user-attachments/assets/6e580840-198a-4c4d-a77c-3e2350ae6652" />

[tfidf_stemmed (1).csv](https://github.com/user-attachments/files/34856451/tfidf_stemmed.1.csv)


````
# Lemmatize edilmiş ürün başlıkları üzerinden TF-IDF hesaplama
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

tfidf_matrix = tfidf.fit_transform(
    df['Lemmatized Title'].fillna('')
)

feature_names = tfidf.get_feature_names_out()

print("İlk ürün başlığında en yüksek TF-IDF skoruna sahip 5 kelime:")

first_row = tfidf_matrix[0].toarray().flatten()
top_indices = first_row.argsort()[-5:][::-1]

for index in top_indices:
    print(f"{feature_names[index]}: {first_row[index]:.4f}")
````
<img width="1493" height="605" alt="Ekran görüntüsü 2026-11-14 194628" src="https://github.com/user-attachments/assets/9bb5efdc-0294-4cae-b40e-a577595c2057" />

````
# Stemmed token listelerini metne dönüştür
stemmed_texts = [
    ' '.join(tokens)
    for tokens in tokenized_corpus_stemmed
]

# TF-IDF vektörizerını başlat
tfidf = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

# Stemmed metinlerden TF-IDF matrisi oluştur
tfidf_matrix_stem = tfidf.fit_transform(stemmed_texts)

# Kelimeleri al
feature_names_stem = tfidf.get_feature_names_out()

print("Stemmed TF-IDF matris boyutu:", tfidf_matrix_stem.shape)

# İlk ürünün en yüksek TF-IDF skoruna sahip 5 kelimesini bul
first_row = tfidf_matrix_stem.getrow(0)

top_indices = first_row.data.argsort()[-5:][::-1]

print("\nİlk ürün başlığında en yüksek TF-IDF skoruna sahip 5 kelime:")

for index in top_indices:
    word_index = first_row.indices[index]
    print(
        f"{feature_names_stem[word_index]}: "
        f"{first_row.data[index]:.4f}"
    )

````
<img width="1487" height="652" alt="Ekran görüntüsü 2026-11-14 194748" src="https://github.com/user-attachments/assets/b015b575-7429-4ea8-bc09-1af0f63d8cc6" />


**Cosine Similarity benzerliğini hesaplıyoruz**

````
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. TF-IDF matrisini oluştur
vectorizer_stem = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_matrix_stem = vectorizer_stem.fit_transform(stemmed_texts)

# 2. Gruplama için değişkenleri hazırla
groups = []
visited = set()
threshold = 0.85

# 3. Benzerlikleri parça parça hesapla
batch_size = 100

for start in range(0, len(stemmed_texts), batch_size):

    end = min(start + batch_size, len(stemmed_texts))

    # Sadece mevcut parçanın benzerliklerini hesapla
    cosine_sim_batch = cosine_similarity(
        tfidf_matrix_stem[start:end],
        tfidf_matrix_stem
    )

    for local_idx in range(end - start):

        i = start + local_idx

        if i in visited:
            continue

        group = [i]
        visited.add(i)

        similar_indices = np.where(
            cosine_sim_batch[local_idx] > threshold
        )[0]

        for j in similar_indices:

            if j > i and j not in visited:
                group.append(j)
                visited.add(j)

        groups.append(group)

    del cosine_sim_batch

# 4. group_id değerlerini oluştur
group_ids = [None] * len(stemmed_texts)

for group_id, group in enumerate(groups):
    for idx in group:
        group_ids[idx] = group_id

# 5. Grup bilgilerini ana veri setine ekle
df['group_id_stemmed'] = group_ids

# 6. Sonuçları kontrol et
print("Toplam grup sayısı:", len(groups))
print("\nİlk 20 ürünün grupları:")

print(
    df[['Product Title', 'group_id_stemmed']].head(20)
)
````
<img width="1483" height="751" alt="Ekran görüntüsü 2026-11-14 194927" src="https://github.com/user-attachments/assets/9cd67b68-a6d7-4250-8b12-0ded26c28cf0" />

````
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. PriceRunner veri setinden bir kelime seç
word = "iphone"

# 2. Kelimenin indeksini bul
feature_names = vectorizer.get_feature_names_out()

if word not in feature_names:
    print(f"'{word}' kelimesi TF-IDF sözlüğünde bulunamadı.")

else:
    word_index = np.where(feature_names == word)[0][0]

    # 3. Seçilen kelimenin TF-IDF vektörünü al
    word_vector = tfidf_matrix.T[word_index]

    print(f"'{word}' kelimesinin TF-IDF vektörü:")
    print(word_vector)

    # 4. Diğer kelimelerle kosinüs benzerliğini hesapla
    similarities = cosine_similarity(
        word_vector,
        tfidf_matrix.T
    ).flatten()

    # 5. En benzer 5 kelimeyi bul
    top_indices = similarities.argsort()[::-1]

    top_indices = [
        idx for idx in top_indices
        if idx != word_index
    ][:5]

    # 6. Sonuçları göster
    print(f"\n'{word}' kelimesine en çok benzeyen 5 kelime:")

    for idx in top_indices:
        print(f"{feature_names[idx]}: {similarities[idx]:.4f}")
````
<img width="1517" height="802" alt="Ekran görüntüsü 2026-11-14 195708" src="https://github.com/user-attachments/assets/b508d5fa-bb9a-44f4-8e6f-5baf63d1ec73" />


**Word2Vec modeline geçiyoruz**

````
# PriceRunner veri setindeki ürün başlıklarını al
texts = df['Product Title'].dropna().astype(str).tolist()

# Ürün başlıklarını tokenize et
tokenized_texts = [
    word_tokenize(text.lower())
    for text in texts
]

# Kontrol
print("Toplam ürün başlığı:", len(texts))

print("\nİlk 5 tokenize edilmiş ürün başlığı:")
for tokens in tokenized_texts[:5]:
    print(tokens)
````
<img width="1487" height="581" alt="Ekran görüntüsü 2026-11-14 195827" src="https://github.com/user-attachments/assets/833606ce-394b-41e8-b55a-0ab48cf2f1e3" />


````
# 1. Tokenized corpus'ları oluştur
tokenized_corpus_lemmatized = tokenized_corpus_lemmatized
tokenized_corpus_stemmed = tokenized_corpus_stemmed

# Boş listeleri çıkar
tokenized_corpus_lemmatized = [
    tokens for tokens in tokenized_corpus_lemmatized if tokens
]

tokenized_corpus_stemmed = [
    tokens for tokens in tokenized_corpus_stemmed if tokens
]

# 2. Parametre setleri
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

# 3. Model eğitme ve kaydetme fonksiyonu
def train_and_save_model(corpus, params, prefix):

    sg_flag = 1 if params['model_type'] == 'skipgram' else 0

    model = Word2Vec(
        sentences=corpus,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=1,
        sg=sg_flag,
        workers=2,
        epochs=10,
        seed=42
    )

    fname = (
        f"{prefix}_{params['model_type']}"
        f"_w{params['window']}_d{params['vector_size']}.model"
    )

    model.save(fname)

    print(f"Model kaydedildi: {fname}")

    # Model belleğini serbest bırak
    del model


# 4. Lemmatized modelleri eğit ve kaydet
for p in parameters:
    train_and_save_model(
        tokenized_corpus_lemmatized,
        p,
        "lemmatized_model"
    )

# 5. Stemmed modelleri eğit ve kaydet
for p in parameters:
    train_and_save_model(
        tokenized_corpus_stemmed,
        p,
        "stemmed_model"
    )
````
<img width="1496" height="747" alt="Ekran görüntüsü 2026-11-14 200138" src="https://github.com/user-attachments/assets/69c89580-8aac-45a8-a1e0-feb548389df5" />


````
# Modelleri yükledik
model_1 = Word2Vec.load("lemmatized_model_cbow_w2_d100.model")
model_2 = Word2Vec.load("stemmed_model_skipgram_w4_d100.model")
model_3 = Word2Vec.load("lemmatized_model_skipgram_w4_d300.model")

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
print_similar_words(model_3, "Lemmatized Skipgram Window 4 Dim 300")
````
<img width="1463" height="688" alt="image" src="https://github.com/user-attachments/assets/6d2ac26d-8d69-489a-935b-3a2f23fce6cc" />


````
# Her model için doğru kelime biçimini kullan
print_similar_words(
    model_1,
    "Lemmatized CBOW Window 2 Dim 100",
    keyword="iphone"
)

print_similar_words(
    model_2,
    "Stemmed Skipgram Window 4 Dim 100",
    keyword="iphon"
)

print_similar_words(
    model_3,
    "Lemmatized Skipgram Window 4 Dim 300",
    keyword="iphone"
)
````
<img width="1490" height="708" alt="Ekran görüntüsü 2026-11-14 200623" src="https://github.com/user-attachments/assets/7d17896e-aad4-40b5-955e-cfe0c5c14b0d" />



# Eğitilen Modellerle Metin Benzerliği Hesaplama ve Değerlendirme
**TF-IDF Benzerliği hesaplama**
````
import pandas as pd

# Dosyayı yükle
df_sample = pd.read_csv("lemmatized_sentences.csv")

# İlk 5 satırı göster
print(df_sample.head())
````


````
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. İlk ürünün TF-IDF vektörünü sorgu olarak al
query_vector = tfidf_matrix[0]

# 2. Tüm ürünlerle cosine similarity hesapla
similarities = cosine_similarity(
    query_vector,
    tfidf_matrix
).flatten()

# 3. En benzer 5 ürünü bul
top_indices = similarities.argsort()[::-1]

# Sorgu ürününün kendisini çıkar
top_indices = [
    idx for idx in top_indices if idx != 0
][:5]

# 4. Sonuçları göster
print("İlk ürün başlığı:")
print(df['Product Title'].iloc[0])

print("\nEn Benzer 5 Ürün (TF-IDF Lemmatized):\n")

for i, idx in enumerate(top_indices, start=1):
    print(f"{i}. Ürün: {df['Product Title'].iloc[idx]}")
    print(f"   Benzerlik Skoru: {similarities[idx]:.4f}")
 ````
<img width="1577" height="657" alt="Ekran görüntüsü 2026-11-14 200937" src="https://github.com/user-attachments/assets/f27ec1d8-3019-40dc-adea-40f38c323294" />


````
# En benzer 5 ürünün skorlarını ve lemmatized metinlerini göster

for i, idx in enumerate(top_indices, start=1):

    print(f"{i}. Benzerlik Skoru: {similarities[idx]:.4f}")

    print(f"   Orijinal Ürün: {df['Product Title'].iloc[idx]}")

    print(f"   Lemmatized Metin: {lemmatized_texts[idx]}\n")
````
<img width="1490" height="623" alt="Ekran görüntüsü 2026-11-14 201046" src="https://github.com/user-attachments/assets/808a132c-8145-4202-91bc-64608eec99c4" />


````
from sklearn.metrics.pairwise import cosine_similarity

# 1. İlk ürünün Stemmed TF-IDF vektörünü sorgu olarak al
query_vector = tfidf_matrix_stem[0]

# 2. Tüm ürünlerle cosine similarity hesapla
similarities_stem = cosine_similarity(
    query_vector,
    tfidf_matrix_stem
).flatten()

# 3. En benzer 5 ürünü bul
top_indices_stem = similarities_stem.argsort()[::-1]

# Sorgu ürününün kendisini çıkar
top_indices_stem = [
    idx for idx in top_indices_stem if idx != 0
][:5]

# 4. Sonuçları yazdır
print("İlk ürün başlığı:")
print(df['Product Title'].iloc[0])

print("\nEn Benzer 5 Ürün (TF-IDF Stemmed):\n")

for i, idx in enumerate(top_indices_stem, start=1):

    print(f"{i}. Benzerlik Skoru: {similarities_stem[idx]:.4f}")

    print(f"   Orijinal Ürün: {df['Product Title'].iloc[idx]}")

    print(f"   Stemmed Metin: {' '.join(tokenized_corpus_stemmed[idx])}\n")
````

<img width="1481" height="738" alt="Ekran görüntüsü 2026-11-14 201213" src="https://github.com/user-attachments/assets/1adedd39-4d94-4edf-b85e-a7a8450734d2" />

````
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
````
````
# Bir ürün başlığındaki kelimelerin vektörlerini alıp ortalamasını hesapla
def get_sentence_vector(model, sentence):

    vectors = []

    for word in sentence.split():

        if word in model.wv:
            vectors.append(model.wv[word])

    if vectors:
        return np.mean(vectors, axis=0)

    else:
        return None
````
````
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# 1. Giriş ürününü belirle
query = df['Product Title'].iloc[0]

print("Giriş ürün başlığı:", query)

# 2. Çalıştırılacak 8 model
model_files = [
    # Lemmatized modeller
    "lemmatized_model_cbow_w2_d100.model",
    "lemmatized_model_skipgram_w2_d100.model",
    "lemmatized_model_cbow_w4_d100.model",
    "lemmatized_model_skipgram_w4_d100.model",

    # Stemmed modeller
    "stemmed_model_cbow_w2_d100.model",
    "stemmed_model_skipgram_w2_d100.model",
    "stemmed_model_cbow_w4_d100.model",
    "stemmed_model_skipgram_w4_d100.model"
]

# 3. Cümleyi vektöre dönüştüren fonksiyon
def get_sentence_vector(model, sentence):

    words = sentence.split()

    word_vecs = [
        model.wv[word]
        for word in words
        if word in model.wv
    ]

    if not word_vecs:
        return None

    return np.mean(word_vecs, axis=0)


# 4. Modelleri sırayla çalıştır
for model_path in model_files:

    print("\n" + "=" * 60)
    print("Model:", model_path)

    # Modeli yükle
    model = Word2Vec.load(model_path)

    # Model türüne göre metin listesini seç
    if "lemmatized" in model_path:
        corpus = tokenized_corpus_lemmatized
        query_tokens = word_tokenize(clean_text(query))

    else:
        corpus = tokenized_corpus_stemmed
        query_tokens = [
            stemmer.stem(word)
            for word in word_tokenize(clean_text(query))
            if word.isalpha()
        ]

    # Giriş metnini vektöre dönüştür
    query_vec = get_sentence_vector(
        model,
        " ".join(query_tokens)
    )

    if query_vec is None:
        print("Giriş metni model sözlüğünde bulunamadı.")
        del model
        continue

    # Ürün başlıklarının vektörlerini hesapla
    sentence_vectors = [
        get_sentence_vector(model, " ".join(tokens))
        for tokens in corpus
    ]

    # Geçerli vektörleri belirle
    valid_indices = [
        i for i, vec in enumerate(sentence_vectors)
        if vec is not None
    ]

    valid_vectors = np.array([
        sentence_vectors[i] for i in valid_indices
    ])

    # Cosine Similarity hesapla
    similarities = cosine_similarity(
        query_vec.reshape(1, -1),
        valid_vectors
    ).flatten()

    # Giriş ürününün kendisini hariç tut
    valid_positions = [
        pos for pos, idx in enumerate(valid_indices)
        if idx != 0
    ]

    # En benzer 5 ürünü bul
    top_positions = sorted(
        valid_positions,
        key=lambda pos: similarities[pos],
        reverse=True
    )[:5]

    # Sonuçları göster
    print("\nEn Benzer 5 Ürün:")

    for rank, position in enumerate(top_positions, start=1):

        idx = valid_indices[position]

        print(f"\n{rank}. Benzerlik Skoru: {similarities[position]:.4f}")
        print("Ürün:", df['Product Title'].iloc[idx])

    # Belleği serbest bırak
    del model, sentence_vectors, valid_vectors, similarities
````
````
MODELİN ÇIKTILARI
Giriş ürün başlığı: apple iphone 8 plus 64gb silver

============================================================
Model: lemmatized_model_cbow_w2_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 1.0000
Ürün: apple iphone 8 plus 64 gb silver

2. Benzerlik Skoru: 0.9953
Ürün: apple iphone 8 plus 256gb silver

3. Benzerlik Skoru: 0.9943
Ürün: apple iphone 8 plus 64gb space grey

4. Benzerlik Skoru: 0.9943
Ürün: apple iphone 8 plus 64 gb space grey

5. Benzerlik Skoru: 0.9943
Ürün: apple iphone 8 plus 64gb space grey

============================================================
Model: lemmatized_model_skipgram_w2_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 1.0000
Ürün: apple iphone 8 plus 64 gb silver

2. Benzerlik Skoru: 0.9956
Ürün: apple iphone 8 plus 256gb silver

3. Benzerlik Skoru: 0.9933
Ürün: apple iphone 8 plus 64gb silver unlocked

4. Benzerlik Skoru: 0.9914
Ürün: apple iphone 8 plus 64 gb spacegrau

5. Benzerlik Skoru: 0.9911
Ürün: apple iphone 8 plus 64 gb rt

============================================================
Model: lemmatized_model_cbow_w4_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 1.0000
Ürün: apple iphone 8 plus 64 gb silver

2. Benzerlik Skoru: 0.9942
Ürün: apple iphone 8 plus 64gb space grey

3. Benzerlik Skoru: 0.9942
Ürün: apple iphone 8 plus 64 gb space grey

4. Benzerlik Skoru: 0.9942
Ürün: apple iphone 8 plus 64gb space grey

5. Benzerlik Skoru: 0.9942
Ürün: apple iphone 8 plus 64gb space grey

============================================================
Model: lemmatized_model_skipgram_w4_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 1.0000
Ürün: apple iphone 8 plus 64 gb silver

2. Benzerlik Skoru: 0.9931
Ürün: apple iphone 8 plus 64gb silver unlocked

3. Benzerlik Skoru: 0.9908
Ürün: apple iphone 8 plus 256gb silver

4. Benzerlik Skoru: 0.9899
Ürün: apple iphone 8 plus 64 gb spacegrau

5. Benzerlik Skoru: 0.9897
Ürün: iphone 8 plus 256gb silver mq8q2b/a wc01

============================================================
Model: stemmed_model_cbow_w2_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 0.9875
Ürün: apple iphone xs 256gb silver

2. Benzerlik Skoru: 0.9875
Ürün: apple iphone xs 256gb silver

3. Benzerlik Skoru: 0.9875
Ürün: apple iphone xs 256gb silver

4. Benzerlik Skoru: 0.9872
Ürün: apple iphone xs max 512 gb silver

5. Benzerlik Skoru: 0.9854
Ürün: apple iphone xs max 256gb silver

============================================================
Model: stemmed_model_skipgram_w2_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 0.9832
Ürün: apple iphone 6s plus

2. Benzerlik Skoru: 0.9824
Ürün: apple iphone 6s plus 64gb silver

3. Benzerlik Skoru: 0.9790
Ürün: apple iphone 7 plus 128gb silver

4. Benzerlik Skoru: 0.9790
Ürün: apple iphone 7 plus 128gb silver

5. Benzerlik Skoru: 0.9758
Ürün: apple iphone 8 plus 256gb silver

============================================================
Model: stemmed_model_cbow_w4_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 0.9885
Ürün: apple iphone xs 256gb silver

2. Benzerlik Skoru: 0.9885
Ürün: apple iphone xs 256gb silver

3. Benzerlik Skoru: 0.9885
Ürün: apple iphone xs 256gb silver

4. Benzerlik Skoru: 0.9852
Ürün: apple iphone xs max 512 gb silver

5. Benzerlik Skoru: 0.9849
Ürün: apple iphone 7 plus 128gb silver

============================================================
Model: stemmed_model_skipgram_w4_d100.model

En Benzer 5 Ürün:

1. Benzerlik Skoru: 0.9804
Ürün: apple iphone 6s plus

2. Benzerlik Skoru: 0.9773
Ürün: apple iphone 6s plus 64gb silver

3. Benzerlik Skoru: 0.9725
Ürün: apple iphone 7 plus 128gb silver

4. Benzerlik Skoru: 0.9725
Ürün: apple iphone 7 plus 128gb silver

5. Benzerlik Skoru: 0.9666
Ürün: apple iphone 8 plus 256gb silver

````
**Sıralama Tutarlılığı Değerlendirmesi (Ranking Agreement)**
**Jaccard benzerliği hesaplama**
````
from gensim.models import Word2Vec
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# 1. Giriş ürününü belirle
query = df['Product Title'].iloc[0]

print("Giriş ürün başlığı:", query)

# 2. Çalıştırılacak 8 Word2Vec modeli
model_files = [
    # Lemmatized modeller
    "lemmatized_model_cbow_w2_d100.model",
    "lemmatized_model_skipgram_w2_d100.model",
    "lemmatized_model_cbow_w4_d100.model",
    "lemmatized_model_skipgram_w4_d100.model",

    # Stemmed modeller
    "stemmed_model_cbow_w2_d100.model",
    "stemmed_model_skipgram_w2_d100.model",
    "stemmed_model_cbow_w4_d100.model",
    "stemmed_model_skipgram_w4_d100.model"
]

# 3. Cümleyi ortalama vektöre dönüştüren fonksiyon
def get_sentence_vector(model, sentence):

    word_vecs = [
        model.wv[word]
        for word in sentence.split()
        if word in model.wv
    ]

    if not word_vecs:
        return None

    return np.mean(word_vecs, axis=0)


# 4. Her modelin ilk 5 sonucunu saklayacağımız sözlük
model_top_indices = {}

# 5. Modelleri sırayla çalıştır
for model_path in model_files:

    print(f"\nİşleniyor: {model_path}")

    model = Word2Vec.load(model_path)

    # Model türüne göre uygun metin listesini seç
    if "lemmatized" in model_path:

        corpus = tokenized_corpus_lemmatized
        query_tokens = word_tokenize(clean_text(query))

    else:

        corpus = tokenized_corpus_stemmed
        query_tokens = [
            stemmer.stem(word)
            for word in word_tokenize(clean_text(query))
            if word.isalpha()
        ]

    # Giriş ürününün vektörünü oluştur
    query_vec = get_sentence_vector(
        model,
        " ".join(query_tokens)
    )

    if query_vec is None:
        print("Giriş ürünü model tarafından temsil edilemedi.")
        del model
        continue

    # Tüm ürünlerin vektörlerini oluştur
    sentence_vectors = [
        get_sentence_vector(model, " ".join(tokens))
        for tokens in corpus
    ]

    # Geçerli vektörleri belirle
    valid_indices = [
        i for i, vec in enumerate(sentence_vectors)
        if vec is not None and i != 0
    ]

    valid_vectors = np.array([
        sentence_vectors[i] for i in valid_indices
    ])

    # Cosine Similarity hesapla
    similarities = cosine_similarity(
        query_vec.reshape(1, -1),
        valid_vectors
    ).flatten()

    # En benzer 5 ürünü bul
    top_positions = np.argsort(similarities)[::-1][:5]

    top_indices = [
        valid_indices[position]
        for position in top_positions
    ]

    # Sonuçları sakla
    model_top_indices[model_path] = top_indices

    print("İlk 5 ürünün indeksleri:", top_indices)

    del model, sentence_vectors, valid_vectors, similarities


# 6. Model sonuçlarını JSON dosyasına kaydet
with open("model_top_indices.json", "w") as f:
    json.dump(model_top_indices, f, indent=2)

print("\nTüm modellerin sonuçları kaydedildi.")
````

[model_top_indices (1).json](https://github.com/user-attachments/files/34857082/model_top_indices.1.json)

````
MODELİN ÇIKTILARI
Giriş ürün başlığı: apple iphone 8 plus 64gb silver

İşleniyor: lemmatized_model_cbow_w2_d100.model
İlk 5 ürünün indeksleri: [15, 179, 3, 6, 7]

İşleniyor: lemmatized_model_skipgram_w2_d100.model
İlk 5 ürünün indeksleri: [15, 179, 16, 1, 916]

İşleniyor: lemmatized_model_cbow_w4_d100.model
İlk 5 ürünün indeksleri: [15, 6, 3, 9, 8]

İşleniyor: lemmatized_model_skipgram_w4_d100.model
İlk 5 ürünün indeksleri: [15, 16, 179, 1, 188]

İşleniyor: stemmed_model_cbow_w2_d100.model
İlk 5 ürünün indeksleri: [903, 895, 902, 985, 952]

İşleniyor: stemmed_model_skipgram_w2_d100.model
İlk 5 ürünün indeksleri: [289, 2845, 233, 227, 179]

İşleniyor: stemmed_model_cbow_w4_d100.model
İlk 5 ürünün indeksleri: [895, 903, 902, 985, 227]

İşleniyor: stemmed_model_skipgram_w4_d100.model
İlk 5 ürünün indeksleri: [289, 2845, 227, 233, 179]

Tüm modellerin sonuçları kaydedildi.
````



**Jaccard Benzerlik Matrisi Hesaplama**
````
import json
import pandas as pd

# 1. JSON dosyasından model sonuçlarını yükle
with open("model_top_indices.json", "r") as f:
    model_indices = json.load(f)

# 2. Model isimlerini al
model_names = list(model_indices.keys())

# 3. Boş Jaccard benzerlik matrisi oluştur
jaccard_matrix = pd.DataFrame(
    0.0,
    index=model_names,
    columns=model_names
)

# 4. Modeller arasındaki Jaccard benzerliklerini hesapla
for model_a in model_names:

    set_a = set(model_indices[model_a])

    for model_b in model_names:

        set_b = set(model_indices[model_b])

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        jaccard_score = (
            intersection / union if union != 0 else 0
        )

        jaccard_matrix.loc[model_a, model_b] = round(
            jaccard_score, 2
        )

# 5. CSV dosyasına kaydet
jaccard_matrix.to_csv(
    "jaccard_similarity_matrix.csv",
    encoding="utf-8-sig"
)

# 6. Sonuçları görüntüle
print("Modeller Arası Jaccard Benzerlik Matrisi:\n")
print(jaccard_matrix)

print("\nJaccard matrisi CSV dosyasına kaydedildi.")
````

[jaccard_similarity_matrix (2).csv](https://github.com/user-attachments/files/34857106/jaccard_similarity_matrix.2.csv)

````
Modeller Arası Jaccard Benzerlik Matrisi:

                                         lemmatized_model_cbow_w2_d100.model  \
lemmatized_model_cbow_w2_d100.model                                     1.00   
lemmatized_model_skipgram_w2_d100.model                                 0.25   
lemmatized_model_cbow_w4_d100.model                                     0.43   
lemmatized_model_skipgram_w4_d100.model                                 0.25   
stemmed_model_cbow_w2_d100.model                                        0.00   
stemmed_model_skipgram_w2_d100.model                                    0.11   
stemmed_model_cbow_w4_d100.model                                        0.00   
stemmed_model_skipgram_w4_d100.model                                    0.11   

                                         lemmatized_model_skipgram_w2_d100.model  \
lemmatized_model_cbow_w2_d100.model                                         0.25   
lemmatized_model_skipgram_w2_d100.model                                     1.00   
lemmatized_model_cbow_w4_d100.model                                         0.11   
lemmatized_model_skipgram_w4_d100.model                                     0.67   
stemmed_model_cbow_w2_d100.model                                            0.00   
stemmed_model_skipgram_w2_d100.model                                        0.11   
stemmed_model_cbow_w4_d100.model                                            0.00   
stemmed_model_skipgram_w4_d100.model                                        0.11   

                                         lemmatized_model_cbow_w4_d100.model  \
lemmatized_model_cbow_w2_d100.model                                     0.43   
lemmatized_model_skipgram_w2_d100.model                                 0.11   
lemmatized_model_cbow_w4_d100.model                                     1.00   
lemmatized_model_skipgram_w4_d100.model                                 0.11   
stemmed_model_cbow_w2_d100.model                                        0.00   
stemmed_model_skipgram_w2_d100.model                                    0.00   
stemmed_model_cbow_w4_d100.model                                        0.00   
stemmed_model_skipgram_w4_d100.model                                    0.00   

                                         lemmatized_model_skipgram_w4_d100.model  \
lemmatized_model_cbow_w2_d100.model                                         0.25   
lemmatized_model_skipgram_w2_d100.model                                     0.67   
lemmatized_model_cbow_w4_d100.model                                         0.11   
lemmatized_model_skipgram_w4_d100.model                                     1.00   
stemmed_model_cbow_w2_d100.model                                            0.00   
stemmed_model_skipgram_w2_d100.model                                        0.11   
stemmed_model_cbow_w4_d100.model                                            0.00   
stemmed_model_skipgram_w4_d100.model                                        0.11   

                                         stemmed_model_cbow_w2_d100.model  \
lemmatized_model_cbow_w2_d100.model                                  0.00   
lemmatized_model_skipgram_w2_d100.model                              0.00   
lemmatized_model_cbow_w4_d100.model                                  0.00   
lemmatized_model_skipgram_w4_d100.model                              0.00   
stemmed_model_cbow_w2_d100.model                                     1.00   
stemmed_model_skipgram_w2_d100.model                                 0.00   
stemmed_model_cbow_w4_d100.model                                     0.67   
stemmed_model_skipgram_w4_d100.model                                 0.00   

                                         stemmed_model_skipgram_w2_d100.model  \
lemmatized_model_cbow_w2_d100.model                                      0.11   
lemmatized_model_skipgram_w2_d100.model                                  0.11   
lemmatized_model_cbow_w4_d100.model                                      0.00   
lemmatized_model_skipgram_w4_d100.model                                  0.11   
stemmed_model_cbow_w2_d100.model                                         0.00   
stemmed_model_skipgram_w2_d100.model                                     1.00   
stemmed_model_cbow_w4_d100.model                                         0.11   
stemmed_model_skipgram_w4_d100.model                                     1.00   

                                         stemmed_model_cbow_w4_d100.model  \
lemmatized_model_cbow_w2_d100.model                                  0.00   
lemmatized_model_skipgram_w2_d100.model                              0.00   
lemmatized_model_cbow_w4_d100.model                                  0.00   
lemmatized_model_skipgram_w4_d100.model                              0.00   
stemmed_model_cbow_w2_d100.model                                     0.67   
stemmed_model_skipgram_w2_d100.model                                 0.11   
stemmed_model_cbow_w4_d100.model                                     1.00   
stemmed_model_skipgram_w4_d100.model                                 0.11   

                                         stemmed_model_skipgram_w4_d100.model  
lemmatized_model_cbow_w2_d100.model                                      0.11  
lemmatized_model_skipgram_w2_d100.model                                  0.11  
lemmatized_model_cbow_w4_d100.model                                      0.00  
lemmatized_model_skipgram_w4_d100.model                                  0.11  
stemmed_model_cbow_w2_d100.model                                         0.00  
stemmed_model_skipgram_w2_d100.model                                     1.00  
stemmed_model_cbow_w4_d100.model                                         0.11  
stemmed_model_skipgram_w4_d100.model                                     1.00  

Jaccard matrisi CSV dosyasına kaydedildi.
````

**Sonuçların Değerlendirilmesi**

````

1. Veri setinin incelenmesi ve metin ön işleme sonuçları


Bu çalışmada, e-ticaret platformlarında aynı ürüne ait farklı satıcı başlıklarının tespit edilebilmesi amacıyla PriceRunner veri seti kullanılmıştır. Veri seti 35.311 ürün kaydı ve 7 sütundan oluşmaktadır. Ürün başlıkları, modelin temel girdisi olarak değerlendirilmiş; ürünlere ait Cluster ID ve Cluster Label bilgileri ise veri setinin orijinal yapısında korunmuştur.

İlk aşamada ürün başlıkları incelenmiş ve metinler üzerinde küçük harfe dönüştürme, HTML etiketlerini temizleme, noktalama işaretlerini düzenleme ve gereksiz boşlukları kaldırma işlemleri gerçekleştirilmiştir.

Ardından metinler üzerinde Tokenization, Lemmatization ve Stemming işlemleri uygulanmıştır. Bu işlemler sonucunda ürün başlıklarının farklı metinsel gösterimleri elde edilmiştir.

Örneğin, apple iphone 8 plus 64gb silver başlığı Lemmatization sonrasında aynı biçimini korurken, Stemming işleminden sonra appl iphon 8 plu 64gb silver biçimine dönüşmüştür.

Bu aşamada Lemmatization işleminin kelimelerin temel biçimini koruduğu, Stemming işleminin ise kelimeleri köklerine indirgerken daha fazla biçimsel değişiklik oluşturduğu gözlemlenmiştir.

2. Zipf Yasası analizi


Ham ürün başlıkları, Lemmatized ve Stemmed metinler üzerinde ayrı ayrı Zipf Yasası analizleri gerçekleştirilmiştir.

Bu analiz ile veri setinde kelimelerin kullanım sıklıkları incelenmiş ve kelime sıralaması ile frekans arasındaki ilişki log-log grafikler üzerinden gösterilmiştir.

Üç farklı metin gösterimi üzerinden analiz yapılması, metin ön işleme işlemlerinin kelime dağılımına nasıl yansıdığını incelemeyi sağlamıştır.

3. TF-IDF sonuçlarının değerlendirilmesi


TF-IDF yöntemi hem Lemmatized hem de Stemmed metinler üzerinde uygulanmıştır.

Lemmatized TF-IDF matrisi 35.311 × 19.157, Stemmed TF-IDF matrisi ise 35.311 × 19.157 boyutunda oluşturulmuştur. Her iki yöntemde de 19.157 farklı özellik elde edilmiştir.

TF-IDF yöntemiyle ürün başlıklarında yer alan kelimelerin, ilgili ürün açısından ayırt edicilik değerleri hesaplanmıştır.

İlk ürün olarak apple iphone 8 plus 64gb silver başlığı seçilmiş ve diğer ürünlerle cosine similarity değerleri hesaplanmıştır.

Lemmatized TF-IDF sonuçları
Benzer ürün	Cosine similarity
apple iphone 8 plus 64 gb silver	1.0000
apple iphone 8 plus 64gb silver unlocked	0.9007
apple iphone 8 plus	0.8466
apple iphone 8 plus 5.5 single sim 4g 64gb silver	0.8219
apple iphone 8 plus 64gb gold smartphone	0.8074

Stemmed TF-IDF sonuçları
Benzer ürün	Cosine similarity
apple iphone 8 plus 64 gb silver	1.0000
apple iphone 8 plus 64gb silver unlocked	0.9277
apple iphone 8 plus	0.8466
apple iphone 8 plus 5.5 single sim 4g 64gb silver	0.8219
apple iphone 8 plus silver 5.5 64gb 4g unlocked sim free	0.8171

Sonuçlar incelendiğinde her iki yöntemin de giriş ürününe oldukça yakın ürünleri üst sıralara taşıdığı görülmüştür. Stemmed TF-IDF yönteminde bazı ürünlerin benzerlik skorlarının daha yüksek olduğu, Lemmatized TF-IDF yönteminde ise farklı ürün başlıklarının sıralamaya dahil edildiği görülmüştür.

Bu sonuçlar, metin ön işleme yönteminin TF-IDF tabanlı benzerlik hesaplamalarını etkilediğini göstermektedir.

4. Word2Vec modellerinin değerlendirilmesi


Çalışmanın sonraki aşamasında kelimeler arasındaki bağlamsal ilişkileri incelemek amacıyla Word2Vec yöntemi kullanılmıştır.

CBOW ve Skip-gram olmak üzere iki farklı eğitim yaklaşımı değerlendirilmiştir. Ayrıca Window değerleri 2 ve 4, vektör boyutları ise 100 ve 300 olarak belirlenmiştir.

Lemmatized ve Stemmed metinler üzerinde ayrı ayrı eğitim gerçekleştirilerek toplam 16 Word2Vec modeli oluşturulmuştur.

Kelime benzerliği incelemesinde iphone ve iphon kelimeleri üzerinden örnek sorgular gerçekleştirilmiştir.

Örneğin Lemmatized Skip-gram Window 4, 300 boyutlu modelde iphone kelimesine en yakın kelimeler arasında apple, 6s ve mnqq2b yer almıştır.

Stemmed Skip-gram Window 4, 100 boyutlu modelde ise iphon kelimesine en yakın kelimeler appl, mnqq2b ve xs olarak belirlenmiştir.

Bu sonuçlar, Word2Vec modellerinin ürün adları, marka isimleri ve model kodları arasında bağlamsal ilişkiler öğrenebildiğini göstermektedir. Bununla birlikte, bazı modellerde elde edilen benzer kelimelerin doğrudan anlamsal olarak ilişkili olmadığı da görülmüştür.
5. Word2Vec ile ürün benzerliği sonuçları


Word2Vec modellerinin ürün başlıklarını karşılaştırma performansını incelemek amacıyla her ürünün kelime vektörlerinin ortalaması alınarak cümle vektörleri oluşturulmuştur.

İlk ürün başlığı olan apple iphone 8 plus 64gb silver sorgu olarak belirlenmiş ve sekiz farklı model üzerinden en benzer beş ürün araştırılmıştır.

Lemmatized modellerin sonuçlarında, giriş ürünüyle aynı ürün ailesine ait farklı satıcı başlıklarının üst sıralarda yer aldığı görülmüştür.

Örneğin Lemmatized CBOW Window 2 modelinde ilk beş sonuç arasında:
- apple iphone 8 plus 64gb silver
- apple iphone 8 plus 256gb silver
- apple iphone 8 plus 64gb space grey
gibi ürünler bulunmaktadır.

Stemmed modellerde ise farklı ürün indeksleri üst sıralara çıkmıştır. Bu durum, Stemming işleminin kelime temsillerini değiştirmesinin modelin benzerlik sonuçlarına da yansıdığını göstermektedir.

Dolayısıyla Word2Vec sonuçlarında yalnızca model mimarisinin değil, metin ön işleme yönteminin de önemli bir etkisi olduğu gözlemlenmiştir.

6. Jaccard benzerliği ile sıralama tutarlılığı değerlendirmesi


Sekiz farklı Word2Vec modelinin ilk beş ürün sonucu karşılaştırılarak Jaccard benzerlik matrisi oluşturulmuştur.

Jaccard benzerliği, iki modelin ilk beş sonuç listesinde bulunan ortak ürünlerin, toplam farklı ürünlere oranını hesaplamak amacıyla kullanılmıştır.

Sonuçlarda özellikle Lemmatized modeller arasında farklı düzeylerde benzerlikler elde edilmiştir.

Örneğin:
- Lemmatized CBOW Window 2 ile Lemmatized CBOW Window 4 modelleri arasında 0.43 Jaccard benzerliği elde edilmiştir.
- Lemmatized Skip-gram Window 2 ile Lemmatized Skip-gram Window 4 modelleri arasında 0.67 benzerlik görülmüştür.
- Stemmed Skip-gram Window 2 ve Window 4 modelleri arasında ise 0.67 benzerlik elde edilmiştir.
- Lemmatized ve Stemmed CBOW modelleri arasında bazı karşılaştırmalarda 0.00 benzerlik görülmüştür.
Bu sonuçlar, farklı Word2Vec modellerinin aynı ürün sorgusu için farklı sonuç kümeleri oluşturabildiğini göstermektedir. Özellikle aynı model mimarisinde Window değerinin değiştirilmesi, sonuçların bir kısmının korunmasına rağmen sıralamaya giren ürünleri de değiştirebilmektedir.

Jaccard değerlerinin yüksek olması, modellerin aynı ürünleri sonuç listelerine dahil ettiğini; düşük olması ise sonuç kümelerinin birbirinden farklılaştığını göstermektedir.

Burada önemli bir nokta, Jaccard yönteminin sıralamadaki pozisyonları değil, yalnızca ortak ürünleri değerlendirmesidir.

7. Genel sonuç ve değerlendirme


Bu çalışma kapsamında e-ticaret platformlarında aynı ürüne ait farklı ürün başlıklarının metin madenciliği ve doğal dil işleme yöntemleriyle karşılaştırılması amaçlanmıştır.

Çalışmada TF-IDF ve Word2Vec yöntemleri kullanılarak ürün başlıklarının farklı biçimlerde temsil edilmesi sağlanmış, cosine similarity yöntemiyle ürünler arasındaki benzerlikler hesaplanmıştır.

TF-IDF yöntemi, ürün başlıklarında bulunan kelimelerin ayırt edicilik değerlerini dikkate alarak benzer ürünleri belirlemiştir. Word2Vec yöntemi ise kelimeler arasındaki bağlamsal ilişkileri öğrenerek ürün başlıklarını vektörler aracılığıyla temsil etmiştir.

Lemmatization ve Stemming işlemlerinin farklı sonuçlar oluşturduğu, aynı şekilde CBOW ve Skip-gram mimarileri ile Window parametresinin de model çıktıları üzerinde etkili olduğu gözlemlenmiştir.

Jaccard benzerliği kullanılarak modellerin ilk beş sonuç kümesi karşılaştırılmış ve modeller arasındaki tutarlılık farklılıkları incelenmiştir.

Genel olarak çalışma, farklı metin ön işleme ve kelime temsil yöntemlerinin ürün benzerliği sonuçlarını nasıl etkilediğini ortaya koymuştur. Elde edilen bulgular, ürün başlıklarının otomatik olarak karşılaştırılması ve benzer ürünlerin tespit edilmesi süreçlerinde doğal dil işleme yöntemlerinin kullanılabileceğini göstermektedir.

````


















  


