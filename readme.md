Bitcoin Trading Strategies Simulator
Proje Amacı
Bu uygulama, Binance API ile Bitcoin fiyatını düzenli olarak çekerek, popüler ve güvenli 5 farklı borsa stratejisini eş zamanlı olarak sanal ortamda test etmeyi ve karşılaştırmalı analiz yapmayı amaçlar. Her strateji 10.000 TL'lik kendi bakiyesi ile işlem yapar ve sonuçlar detaylı grafikler ile kar/zarar tabloları halinde sunulur.

Özellikler
Binance API anahtarı ile 5 dakikada bir Bitcoin fiyatı çekme

Kısa sürede en çok kullanılan ve güvenli 5 farklı trade stratejisinin uygulanması

Her strateji için 10.000 TL'lik bağımsız sanal bakiye ile long/short işlemlerini simüle etme
Gerçek zamanlı al-sat simülasyonunda sinyal gücüne göre pozisyon büyüklüğü ayarlanır
- İsteğe bağlı "tüm bakiye" modu her al-sat sinyalinde mevcut bakiyenin tamamını kullanır
- Beklenen kâr %2'yi aşıyorsa sinyal gücü düşük olsa bile tüm pozisyon satılabilir
- Her işlemde komisyon ve slipaj maliyeti simüle edilebilir
- Kazanan işlemler arttıkça pozisyon büyüklüğü otomatik olarak artar
- Volatilite ve sinyal gücüne göre dinamik kar al ve trailing-stop seviyeleri
- EMA kesişimi ve hacim kırılımı ilk küçük pozisyonu tetikler, trend
  onaylandıkça kademeli alım yapılır
- Kaçan fırsatları yakalamak için eşik otomatik düşürülür

Her strateji için ayrı grafik:

Fiyat eğrisi

Alım noktaları (kırmızı), satım noktaları (yeşil)

Toplu kar/zarar tablosu:

Her stratejinin toplamda ne kadar kazandırdığı/kaybettirdiği
Her stratejinin elinde kalan BTC miktarı ve karşılık gelen değeri
Her stratejinin simülasyon sonundaki toplam bakiyesi
Güncellenmiş bakiye ile her işlemi listeleyen ayrıntılı işlem günlüğü

Grafikler ve tablo arası kolay geçiş

Modern ve kullanıcı dostu arayüz

Kurulum
Python 3.10+ yüklü olmalı

Gerekli kütüphaneleri yükleyin:
```
pip install -r requirements.txt
```
Proje arayüz için `tkinter` ve grafikler için `matplotlib` paketlerini
kullanmaktadır. Gerekiyorsa aşağıdaki komutla kurabilirsiniz:
```
pip install matplotlib
```
Binance API Key ve Secret’ınızı settings.py veya .env dosyasına ekleyin.

Kullanım
Uygulamayı başlatın:
```
python main.py
```
Uygulama arayüzünde, stratejilerin performansını ve grafiklerini inceleyin.

Kar/zarar tablosu ikonuna tıklayarak, tüm stratejilerin finansal özetini görün.

Proje Mimarisi
services/data_service.py : Binance’tan fiyat verisi çeker

strategies/ : Her bir trade stratejisi için ayrı Python modülü

services/simulation.py : Al/Sat işlemlerini ve bakiye simülasyonunu yürütür

services/logger.py : Yapılan işlemleri ve strateji performansını kaydeder

main.py : Uygulamanın ana akışı ve arayüz yönetimi

Stratejiler
Başlangıçta aşağıdaki örnek stratejiler kullanılacaktır:

RSI (Relative Strength Index)

MACD (Moving Average Convergence Divergence)

Bollinger Bands

MA Cross (Moving Average Cross)

Custom (Kullanıcıya özgü veya sonradan eklenebilir strateji)

Dynamic Hybrid (ATR & hacim filtresi, uyarlanabilir risk yönetimi, seans
eşikleri, piyasa rejimi algısı, çoklu zaman dilimi trend filtreleri, komisyon/slipaj simülasyonu ve parametre optimizasyonu)
kaçan fırsatları tespit ederek eşiği dinamik düşürür

Her stratejinin kendi kuralları ile işlemleri tetiklenir ve sonuçlar görselleştirilir.

Geliştirme Planı
Strateji algoritmaları için yeni modüller eklenebilir

Arayüzde ek filtreler, indikatörler ve performans analiz panelleri geliştirilecek

Gerçek trade ortamına entegrasyon ve uyarı sistemi planlanıyor

Katkı ve İletişim
Geliştirmeye katkı sağlamak veya öneride bulunmak için lütfen pull request açın ya da iletişime geçin.

Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
Bu belgenin Ingilizce versiyonu için [README_EN.md](README_EN.md) dosyasına bakabilirsiniz.

### Parametre Ayarlama Örneği
```python
from strategies.dynamic_hybrid import DynamicHybridStrategy
from services.data_service import DataService

prices = DataService().get_historical_prices(limit=500)
strategy = DynamicHybridStrategy()
grid = {"base_threshold": [0.1, 0.15], "lookback": [40, 60]}
best = strategy.optimize_by_regime(prices, grid)
print(best)
```
