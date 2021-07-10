# flowers-classification-with-cnn
## Veriseti Tanıtımı:
Veri seti, 3670 adet görüntü ve daisy, dandelion, roses, sunflowers, tulips olmak üzere 5 farklı sınıftan 
oluşmaktadır. Daisy sınıfında 633, Dandelion 898, Roses 641, Sunflowers 699 ve Tulips 799 adet 
görüntü içermektedir. Sınıflara ait etiketleme işlem yapılarak daisy sınıfına 0, dandelion sınıfına 1, 
roses sınıfına 2, sunflowers sınıfına 3 ve tulips sınıfına 4 etiketi verildi.
Veri seti örnekleri aşağıdaki gibidir.
![alt text](https://github.com/htcoztrk/flowers-classification-with-cnn/blob/master/flowers_exmple.PNG "Logo Title Text 1")

#### Model Oluşturma
<ul>
<li>Sınıflandırma işlemi için CNN modeli oluşturuldu.
Birçok Convolutional Katmanı arka arkaya kondu ve her birinden sonra ReLU katmanı eklendi. Ve 
bundan sonra Pooling katmanları ve Flattening katmanı eklendi . Daha sonra ReLu katmanı kadar Fully Connceted katmanı eklendi.</li>
<li>Overfitting’i azaltmak için bir teknik olan, bir düzenlenme biçimi olan Dropout katmanı modele 
eklendi. Dropout'u bir katmana uyguladığınızda, eğitim işlemi sırasında katmandan rastgele bir dizi 
çıktı birimini (etkinleştirmeyi sıfıra ayarlayarak) çıkarır. Bu işlem ile, katmandan rastgele çıktı 
birimlerinin % 20'si çıkarıldı.</li>
<li>Aktivasyon “softmax” dır. Softmax, çıkışın 1’e kadar çıkmasını sağlar, böylece çıkış sinyali olasılık 
olarak yorumlanabilir. Daha sonra model, hangi seçeneğin en yüksek olasılıklara sahip olduğuna bağlı 
olarak tahminini yapacaktır.</li>
</ul>

#### Convolutional Layer:
Özellikleri saptanmak için kullanıldı. Resmin özelliklerini algılamaktan 
sorumludur. Bu katman, görüntüdeki düşük ve yüksek seviyeli özellikleri çıkarmak için resme bazı 
fitreler uygular.

#### Pooling (Downsampling) Layer: 
Ağırlık sayısını azaltır ve uygunluğu kontrol eder. Bu katmanın görevi, 
gösterimin kayma boyutunu ve ağ içindeki parametreleri ve hesaplama sayısını azaltmak içindir. Bu 
sayede ağdaki uyumsuzluk kontrol edilmiş olur. Birçok Pooling işlemleri vardır, bu modelde en popüler 
olan max pooling kullanıldı.
#### Flattening Layer : 
Bu katman Klasik Sinir Ağı için verileri hazırlar. 
#### Fully-Connected Layer : Bu katman modelin son ve en önemli katmanıdır. Verileri Flattening 
işleminden alır ve Sinir ağı yoluyla öğrenme işlemini geçekleştirir.

Model eğitimi için kullanılan veri seti %80-%20 oranında train ve test verisi olarak bölündü ve 
Epoch sayısı 20 olarak belirlendi.
## Confusion Matrix
![alt text](https://github.com/htcoztrk/flowers-classification-with-cnn/blob/master/confusion_matrix.PNG "Logo Title Text 1")





