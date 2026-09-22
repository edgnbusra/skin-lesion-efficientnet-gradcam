# Cilt Lezyonu Sınıflandırma - EfficientNet-B3 + Grad-CAM

## Amaç

Bu projede iki temel hedef vardı:
1. **EfficientNet-B3 fine-tuning**: Önceden eğitilmiş (pretrained) bir EfficientNet-B3 modelini, cilt lezyonu görüntülerini sınıflandıracak şekilde transfer learning ile uyarlamak; training loop ve early stopping mantığını sıfırdan yazarak öğrenmek.
2. **Grad-CAM ile açıklanabilirlik**: Modelin bir görüntüyü sınıflandırırken hangi bölgeye baktığını görselleştirmek (explainable AI).

Veri seti: [HAM10000 (Skin Cancer MNIST)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) - 10.015 dermatoskopik cilt lezyonu görüntüsü, 7 sınıf:

| Kod | Anlamı |
|---|---|
| akiec | Aktinik keratoz |
| bcc | Bazal hücreli karsinom |
| bkl | Benign keratoz |
| df | Dermatofibrom |
| mel | Melanom (kötü huylu, en tehlikeli) |
| nv | Melanositik nevüs (normal ben) |
| vasc | Vasküler lezyon |

## Donanım kararı

Yerel makine (Intel i5-13420H, 8GB RAM, dedike GPU yok - sadece Intel UHD Graphics) EfficientNet-B3 fine-tuning için yetersiz bulundu. Bu yüzden:
- **Kod yazımı ve Grad-CAM/değerlendirme (CPU-yeterli işler)** yerel makinede yapıldı.
- **Eğitim (training loop çalıştırma, GPU gerektiren kısım)** Google Colab'in ücretsiz T4 GPU'sunda yapıldı.

## Yapılan işler / Yol haritası

1. **Veri hazırlığı** (`src/dataset.py`, `src/transforms.py`, `src/split_data.py`)
   - HAM10000 metadata'sı okunup PyTorch `Dataset` sınıfına bağlandı.
   - Train/val/test bölünmesi `lesion_id` bazında yapıldı (aynı lezyonun farklı fotoğraflarının farklı split'lere düşüp veri sızıntısına (data leakage) yol açmasını önlemek için `GroupShuffleSplit` kullanıldı).
   - Sonuç: train 7002, val 1519, test 1494 görüntü.
   - Augmentation (random crop, flip, rotation, color jitter) train setine uygulandı.

2. **Model** (`src/model.py`)
   - `torchvision.models.efficientnet_b3` ImageNet ağırlıklarıyla yüklendi.
   - Backbone (özellik çıkarıcı gövde) donduruldu, sadece son sınıflandırma katmanı 7 sınıfa göre yeniden tanımlanıp eğitildi.

3. **Sınıf dengesizliği tespiti ve çözümü**
   - `nv` sınıfı (4718 örnek) ile `df` sınıfı (89 örnek) arasında ~53 kat fark bulundu.
   - `CrossEntropyLoss`'a `compute_class_weight(class_weight="balanced")` ile hesaplanan ağırlıklar eklendi.

4. **Training loop + Early Stopping** (`src/train.py`)
   - Manuel training loop: forward pass -> loss -> backward pass -> optimizer.step().
   - Early stopping: validation loss 5 epoch üst üste iyileşmezse eğitim durduruluyor, en iyi model (`best_model.pth`) ayrıca kaydediliyor.
   - Colab'de (`notebooks/train_colab.ipynb`) T4 GPU ile çalıştırıldı, **early stopping epoch 15'te devreye girdi**.

5. **Grad-CAM** (`src/grad_cam.py`, `src/run_grad_cam.py`)
   - PyTorch forward/backward hook'ları kullanılarak sıfırdan (hazır kütüphane kullanılmadan) implement edildi.
   - Son konvolüsyon katmanının activation'ları ve gradyanları alınıp Global Average Pooling ile kanal önemleri hesaplandı, ısı haritası üretildi.
   - Her sınıftan örnek görüntüler için ısı haritaları `outputs/grad_cam/` klasörüne kaydedildi.

6. **Değerlendirme** (`src/evaluate.py`)
   - Test seti (1494 görüntü) üzerinde confusion matrix ve sınıf bazlı precision/recall/F1 raporu üretildi.

## Sonuçlar

**Genel test accuracy: %67.9**

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| akiec | 0.433 | 0.619 | 0.510 |
| bcc | 0.464 | 0.574 | 0.513 |
| bkl | 0.473 | 0.645 | 0.546 |
| df | 0.091 | 0.571 | 0.157 |
| mel | 0.367 | 0.471 | 0.412 |
| nv | 0.937 | 0.734 | 0.823 |
| vasc | 0.306 | 0.714 | 0.429 |

Detaylar: `outputs/classification_report.txt`, `outputs/confusion_matrix.png`

**En kritik bulgu:** `mel` (melanom, kanser) sınıfının recall'u sadece %47.1 - gerçek melanom vakalarının yarısından fazlası kaçırılıyor, bunların 37 tanesi yanlışlıkla "normal ben" (`nv`) olarak sınıflandırılmış. Bu, tıbbi bir uygulamada en tehlikeli hata türlerinden biri ve modelin mevcut haliyle (sadece son katman eğitilmiş, tek fine-tuning turu) production'a hazır olmadığını gösteriyor.

**Grad-CAM gözlemi:** Belirgin lezyon sınırı olan görüntülerde ısı haritası net şekilde lezyonun üzerine odaklanıyor (model doğru bölgeye bakıyor), yanlış tahmin edilen örneklerde bile bu odaklanma görülüyor - yani hata "rastgele" değil, "ilgili bölgeye bakıp yanlış yorumlama" kaynaklı.

## Olası geliştirmeler (yapılmadı, gelecek için not)

- Backbone'un son bloklarını açıp (unfreeze) düşük learning rate ile ikinci bir fine-tuning turu yapmak.
- Daha fazla epoch / farklı learning rate ile deneme.
- Azınlık sınıflar (df, vasc) için oversampling veya ek veri toplama.

## Proje yapısı

```
src/
  dataset.py       - PyTorch Dataset sınıfı
  transforms.py     - Augmentation ve normalize
  split_data.py     - Train/val/test bölme (lesion_id bazlı)
  model.py          - EfficientNet-B3 model tanımı
  train.py          - Training loop + early stopping (yerel test/Colab için)
  grad_cam.py       - Grad-CAM implementasyonu (hook tabanlı)
  run_grad_cam.py   - Grad-CAM görselleştirmelerini üretme script'i
  evaluate.py       - Test seti değerlendirmesi (confusion matrix, classification report)
notebooks/
  train_colab.ipynb - Colab'de GPU ile eğitim için notebook
outputs/
  best_model.pth           - Eğitilmiş model ağırlıkları
  grad_cam/                - Grad-CAM görselleştirmeleri
  confusion_matrix.png
  classification_report.txt
```

## Not

`data/` klasöründeki HAM10000 veri seti (~5GB) bu repoya dahil edilmemiştir (`.gitignore`'da hariç tutulmuştur). Yeniden indirmek için:
```
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```
