# ecommerce-nlp-model
TF-IDF ve Word2Vec ile ürün başlıkları üzerinden benzerlik analizi yaptık

**E-Ticaret Projesi: Ürün Açıklama Eşleştirme**

**Problem nedir:**
-E-ticaret platformlarında farklı satıcılar, aynı ürünü farklı açıklamalarla listeleyebilmektedir. Bu durum, kullanıcı deneyimini olumsuz etkileyebileceği gibi, -ürünlerin doğru şekilde gruplanamamasına da yol açar. Bu projenin amacı, benzer ürün açıklamalarını gruplayarak **ürün kümeleri** oluşturmak ve tekrar eden kayıtları anlamlı bir şekilde bir araya getirmektir.

**Amaç:**
Farklı satıcılara ait açıklamaları analiz ederek, aynı ürünü ifade eden açıklamaları **otomatik olarak gruplayan** bir sistem geliştirmektir.

**Kullanılan Yöntemler ve Adımlar:**

**Veri Kaynağı:**

-Bu projede kullanılan ürün açıklamaları verisi, Kaggle platformundaki Shopee - Price Match Guarantee yarışmasından alınmıştır.
-Kullanılan dosya: train.csv
-İçeriğinde: Ürün başlıkları (title), ürün ID’leri (posting_id), grup kimlikleri (label_group) gibi alanlar yer almaktadır.
-Veri, farklı satıcıların aynı ürünü nasıl farklı şekilde adlandırdığını gözlemlemek ve bu açıklamaları gruplayarak eşleştirme yapmak için kullanılmıştır.
-Açıklamalar, genellikle marka ve fiyat gibi ek unsurlar içerdiğinden, veri ön işleme gereklidir.

**Kaynak:** Kaggle - Shopee Price Match Guarantee yarışması/ https://www.kaggle.com/competitions/shopee-product-matching/data?select=train.csv
- **Dosya:** 'train.csv'
-**Küçültülmüş dosya** 'train_sample_5000.csv'
  
**Zipf Yasası Analizi**

-Kelimelerin frekans dağılımı incelenerek açıklama yapılarının doğallığı ve bilgi yoğunluğu değerlendirilmiştir.

**Veri Temizleme**

- Marka, fiyat, boyut gibi ayırt edici ancak gruplaştırmaya engel olabilecek bilgiler açıklamalardan temizlenmiştir.
  
**Vektörleştirme:**

-Her açıklama, içeriğindeki kelimelerin vektörlerinin TF-IDF ağırlıklı ortalaması alınarak temsil edilmiştir.
-Açıklamalardan kelime temsilleri oluşturmak için Word2Vec modeli eğitilmiştir.
-Böylece her kelimenin açıklamadaki önemi dikkate alınarak daha anlamlı ve ayrım gücü yüksek vektörler elde edilmiştir.


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
  
**Nasıl Çalıştırılır?**

**Öncelikle gerekli kütüphaneleri yüklüyoruz**

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


**Ardından veri setini çekiyoruz**

**Kaynak:** Kaggle - Shopee Price Match Guarantee yarışması/ https://www.kaggle.com/competitions/shopee-product-matching/data?select=train.csv , buradan train.csv verisini çekiyoruz.

**Orijinal veri**
df = pd.read_csv("train.csv")

**5 000 satırlık rastgele alt küme**
df_sample = df.sample(n=5000, random_state=42)

**Yeni dosyayı kaydet**
df_sample.to_csv("train_sample_5000.csv", index=False)

print("Oluşturulan alt küme satır sayısı:", df_sample.shape[0])

**Yeni oluşturduğumuz alt küme dosyasını yükle**
df_sample = pd.read_csv("train_sample_5000.csv")


**Veriyi incele**
print(df_sample.head())

  


