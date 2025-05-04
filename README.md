# ecommerce-nlp-model
TF-IDF ve Word2Vec ile ürün başlıkları üzerinden benzerlik analizi yaptık

**E-Ticaret Projesi: Ürün Açıklama Eşleştirme**

**Problem nedir:**
-E-ticaret platformlarında farklı satıcılar, aynı ürünü farklı açıklamalarla listeleyebilmektedir. Bu durum, kullanıcı deneyimini olumsuz etkileyebileceği gibi, -ürünlerin doğru şekilde gruplanamamasına da yol açar. Bu projenin amacı, benzer ürün açıklamalarını gruplayarak **ürün kümeleri** oluşturmak ve tekrar eden kayıtları anlamlı bir şekilde bir araya getirmektir.
**Amaç:**
-Farklı satıcılara ait açıklamaları analiz ederek, aynı ürünü ifade eden açıklamaları **otomatik olarak gruplayan** bir sistem geliştirmektir.
**Kullanılan Yöntemler ve Adımlar:**
**Veri Kaynağı:**
-Bu projede kullanılan ürün açıklamaları verisi, Kaggle platformundaki Shopee - Price Match Guarantee yarışmasından alınmıştır.
-Kullanılan dosya: train.csv
-İçeriğinde: Ürün başlıkları (title), ürün ID’leri (posting_id), grup kimlikleri (label_group) gibi alanlar yer almaktadır.
-Veri, farklı satıcıların aynı ürünü nasıl farklı şekilde adlandırdığını gözlemlemek ve bu açıklamaları gruplayarak eşleştirme yapmak için kullanılmıştır.
-Açıklamalar, genellikle marka ve fiyat gibi ek unsurlar içerdiğinden, veri ön işleme gereklidir.
**Veri Temizleme**
- Marka, fiyat, boyut gibi ayırt edici ancak gruplaştırmaya engel olabilecek bilgiler açıklamalardan temizlenmiştir.
**Vektörleştirme:**
-Açıklamalardan kelime temsilleri oluşturmak için Word2Vec modeli eğitilmiştir.

-Her açıklama, içeriğindeki kelimelerin vektörlerinin TF-IDF ağırlıklı ortalaması alınarak temsil edilmiştir.

-Böylece her kelimenin açıklamadaki önemi dikkate alınarak daha anlamlı ve ayrım gücü yüksek vektörler elde edilmiştir.
**Benzerlik Ölçümü:**
- **Cosine Similarity** metriği kullanılarak açıklamalar arasındaki benzerlikler hesaplanmıştır.
**Kullanılan Teknolojiler:**
- **Python**: Proje dili
- **Gensim** (Word2Vec): Kelime vektörleri oluşturma
- **Scikit-learn**: Cosine Similarity hesaplama
- **Pandas / NumPy**: Veri işleme ve matematiksel hesaplamalar
- **Jupyter Notebook**: Proje geliştirme ve sunum platformu
**Dosya İçeriği:**
- `ecommerce_nlp_model.ipynb`: Proje adımlarını içeren Jupyter Notebook dosyası
- `README.md`: Bu açıklama dosyası
**Nasıl Çalıştırılır?**

  


