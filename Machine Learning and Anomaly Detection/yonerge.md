**1\. Proje Tanımı**

Bu projenin amacı, ahşap yüzeylerde meydana gelen yüzey hatalarının (defect) tespit edilmesi için denetimsiz (unsupervised) bir öğrenme modeli geliştirmektir.

Verilen veri seti yalnızca "sağlam" (good) örnekleri içeren eğitim verileriyle modelin öğrenmesini, ardından test aşamasında "sağlam" ve "hatalı" örnekleri ayırt edebilmesini hedeflemektedir.

**2\. Veri Seti Bilgileri**

Veri seti aşağıdaki dizin yapısına sahiptir:

Dataset/

└── wood/

├── train/

│ └── good/

└── test/

├── good/

└── defect/

Eğitim Verisi (train/good/): Sadece sağlam (hatasız) ahşap yüzey örneklerini içerir.

Test Verisi (test/): Hem sağlam (good) hem de hatalı (defect) örnekleri içerir.

Not: Veri setinde maskeler (defect bölgelerini gösteren etiketler) bulunmamaktadır. Bu nedenle modelin yalnızca görüntü temelli bir şekilde anomalileri öğrenmesi beklenmektedir.

**3\. Model Gereksinimleri**

Model, unsupervised anomaly detection tekniklerinden biri kullanılarak eğitilmelidir.

Eğitim sırasında yalnızca train/good klasöründeki veriler kullanılacaktır.

Test sırasında model, test klasöründeki good ve defect örneklerini ayrıştırmaya çalışacaktır.

Model, yalnızca hatayı tespit etmekle kalmayıp, hata lokalizasyonu (defect map) da üretmesi beklenmektedir.

**İpucu:** Görselleri modele vermeden önce arka planı çıkarmak (background removal) veya ROI kırpması yapmak model performansını artırabilir.

**4\. Değerlendirme Kriterleri**

Proje değerlendirmesi üç ana ölçüte göre yapılacaktır:

- **F1 Skoru:** Test setindeki _good_ ve _defect_ örneklerinin doğru sınıflandırılma başarımı, modelin genel performansını temsil edecektir.
- **Hata Lokalizasyonu:** Modelin ürettiği anomaly veya heatmap çıktılarının görsel olarak gerçek hatalı bölgeleri ne kadar doğru tespit ettiği değerlendirilecektir.
- **AUC Skoru (ROC-AUC):** Modelin tüm eşik değerleri boyunca _good-defect_ ayrımında genel ayırt ediciliğini gösterir.

Bu üç ölçüt birlikte dikkate alınarak nihai performans puanı belirlenecektir.

**5\. Teslim Şekli**

- Eğitim kodu (.py veya notebook formatında)
- Eğitilmiş model dosyası
- Sonuç görselleri (örnek good/defect + heatmap)
- Teknik rapor (PDF veya Markdown):
  - Son yıllarda çıkmış ve yüksek atıf almış olan modellerin/ilgili çalışmaların teknik özeti
  - Öğrencinin kullandığı yöntemler açıklamaları
  - Eğitim ayarları (batch size, epoch, learning rate, optimizer vs.)
  - Eğitimin toplam süresi (yaklaşık saat ve epoch cinsinden)
  - Inference süresi (bir görüntü başına veya ortalama)
  - Kullanılan donanım (örneğin CPU, GPU modeli, RAM miktarı vb.)
  - Elde edilen F1 ve Auc skorları
  - Gözlemler

İlk ara rapor (%30) 8. Hafta vize haftası:

**Beklentiler:**

Rapor kalitesinin yüksek olması, kullanılan ekran görüntülerinin açıklanabilir ve anlaşılabilir olması

Öğrencinin kapsamlı literatür araştırması yapması ve makaleleri anlayarak özetleyebilmesi. Not: GenAI ile yapılan özetlemeler sıfır puan alacaktır.

Öğrencinin sağlanan verileri programlama ortamında ön işlemelere tabi tutabilmesi ve ideal olarak ön sonuçlar alarak bunları raporlaması

Final raporu (%30)

**Beklentiler:**

Rapor kalitesinin yüksek olması, kullanılan ekran görüntülerinin açıklanabilir ve anlaşılabilir olması

**Öğrencinin en az 3 anomali tespiti** modeli ile denemeler yapıp başarılı bir şekilde sonuç elde edebilmesi.

Kullandığı modellerin ve aldığı sonuçların teknik olarak artı ve eksilerini makeleye atıflarla yorumlayabilmesi.

Model başarı metrikleri (ideal olarak) yüksek sonuçlar elde ederek bunları raporlayabilmesi ve sonuçların tutarlı olarak ikinci bir kişi tarafından tekrar üretilebilmesi (reproducibility)

Dataset link:

<https://drive.google.com/file/d/1bdSa962SUjPBcaAnlAn3FppQxpy7I1Kh/view?usp=sharing>