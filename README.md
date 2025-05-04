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
- Kelimelerin frekans dağılımı incelenerek açıklama yapılarının doğallığı ve bilgi yoğunluğu değerlendirilmiştir.

**Veri Temizleme**
- Marka, fiyat, boyut gibi ayırt edici ancak gruplaştırmaya engel olabilecek bilgiler açıklamalardan temizlenmiştir.
  
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
  
**Nasıl Çalıştırılır?**

  


