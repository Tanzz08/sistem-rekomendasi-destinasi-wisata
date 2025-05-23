# Laporan Proyek Machine Learning - Bustanul Khair Rahman

## Project Overview

Indonesia merupakan negara kepulauan yang memiliki potensi wisata sangat besar dengan keberagaman destinasi, mulai dari wisata alam, budaya, sejarah, hingga wisata modern. Namun, berdasarkan data dari Kementerian Pariwisata, kontribusi sektor pariwisata terhadap PDB Indonesia masih lebih rendah dibandingkan negara tetangga seperti Thailand dan Malaysia (Suryani, 2018). Salah satu tantangan yang dihadapi adalah kurangnya pemerataan kunjungan wisatawan ke berbagai daerah di Indonesia. Bali, sebagai contoh, menyumbang lebih dari 40% kunjungan wisatawan mancanegara, sementara banyak daerah lain masih belum banyak tereksplorasi (Putra & Widodo, 2020).

Di sisi lain, dalam era digital, wisatawan cenderung melakukan pencarian informasi secara daring sebelum merencanakan perjalanan. Sebuah studi oleh Nugroho dan Lestari (2020) menyebutkan bahwa lebih dari 70% wisatawan domestik menggunakan aplikasi atau platform digital untuk mencari destinasi yang sesuai dengan preferensi mereka. Oleh karena itu, keberadaan sistem rekomendasi berbasis data menjadi sangat penting untuk membantu wisatawan menemukan destinasi yang relevan dan personal, serta mendorong pemerataan kunjungan ke berbagai wilayah.

Dengan memanfaatkan pendekatan machine learning, khususnya content-based filtering dan collaborative filtering, sistem rekomendasi dapat memberikan saran destinasi berdasarkan minat pengguna maupun perilaku pengguna lain yang serupa (Resita & Hidayatullah, 2021). Sistem semacam ini juga diharapkan dapat meningkatkan efisiensi dalam perencanaan perjalanan dan memperkuat promosi destinasi wisata lokal secara digital.

## Business Understanding
### Problem Statements

Bagaimana cara merekomendasikan destinasi wisata yang relevan dan sesuai dengan preferensi pengguna untuk meningkatkan pengalaman perjalanan?

### Goals
- Memberikan rekomendasi tempat wisata yang sesuai dengan minat pengguna
- Menyediakan dua pendekatan sistem rekomendasi yaitu content-based filtering dan collaborative filtering
- Meningkatkan relevansi dan personalisasi dalam pencarian destinasi wisata

### Solution statements
- Menerapakan **Content-Based Filtering**: Menggunakan cosine similarity untuk menghitung antar destinasi berdasarkan fitur tempat wisata
- Menerapakan **Collaborative Filtering**: Menggunakan pendekatan deep learning untuk mempelajari pola rating dari pengguna terhadap tempat wisata

## Data Understanding
Pada proyek ini, dataset yang digunakan berasal dari penyedia open dataset dari situs [Kaggle-Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data). Dataset ini berisi sejumlah tempat wisata di 5 kota besar di Indonesia, yaitu Jakarta, Yogyakarta, Semarang, Bandung, dan  Surabaya.

**Terdapat 4 file pada dataset ini, yaitu**:
- **tourism_with_id.csv**: berisi informasi mengenai tempat-tempat wisata di 5 kota besar di Indonesia dengan total sekitar 400 entri.
- **user.csv**: yang berisi data pengguna dummy yang digunakan untuk membangun fitur sistem rekomendasi berbasis pengguna.
- **tourism_rating.csv**: yang terdiri dari 3 kolom, yaitu pengguna, tempat wisata, dan rating ayng diberikan; berfungsi untuk membangun sistem rekomendasi berbasis rating
- **package_tourism.csv**: yang berisi rekomendasi tempat wisata terdekat berdasarkan waktu, biaya, dan rating. Namun kita tidak menggunakan data ini karena kita berfokus pada rekomendasi berdasarkan tempat yang disukai dan berbasis rating

**Variabel-variabel pada dataset adalah sebagai berikut**:
1. tourism_rating.csv:
   - User_Id : id user
   - Place_Id : id tempat wisata
   - Place_Ratings : rating tempat wisata
2. tourism_with_id.csv:
   - Place_Id : id tempat wisata
   - Place_Name : nama tempat wisata
   - Description : deskripsi dari tempat wisata
   - City : kota tempat wisata berada
   - Price : Harga destinasi wisata
   - Rating : rating dari tempat wisata
   - Time_Minutes : destinasi waktu tempat wisata dalam menit
   - Coordinate : kordinat tempat wisata
   - Lat : latitude dari tempat wisata
   - Long : longitude dari tempat wisata
   - Unnamed: 11 : tak terdefenisi
   - Unnamed: 12 : tak terdefenisi
3. user.csv:
   - User_Id : id user
   - Location : lokasi pengguna
   - Age : usia pengguna
4. package_tourism.csv:
   - Pakcage : paket destinasi wisata. ini adalah primary key
   - City : Kota tempat wisata berada
   - Place_Tourism1 : Tempat wisata pertama
   - Place_Tourism2 : Tempat wisata Kedua
   - Place_Tourism3 : Tempat wisata Ketiga
   - Place_Tourism4 : Tempat wisata Keempat
   - Place_Tourism5 : Tempat Wisata Kelima

   **Kondisi Dataset**:
   1. tourism_rating.csv: 
      - Jumlah data = 10000 entries
      - Missing Value = 0
      - Duplikat = 0 
   2. tourism_with_id.csv:
      - Jumlah data = 437 entries
      - Missing Value : 
         - Time_Minutes = 232
         - Unnamed: 11 = 437
      - Duplikat = 0 



### Exploratory Data Analysis
1. Melihat Count Traveler berdasarkan Usia
   ![Screenshot 1](img/count%20travel%20age.png)

   Max age traveler adalah usia 30 tahun. dan di lanjut dengan usia gen z hingga milenial
3. Melihat Distribusi Lokasi User
   ![Screenshot 2](img/distribusi%20lokasi.png)

   Sebgaian besar user berlokasi di Bekasi, Semarang, Cirebon, dan Yogyakarta
5. Melihat Correlation Matrix dari penggabungan data tourism rating dan tourism with id
   ![Screenshot 3](img/corr%20matrix.png)

   terlihat beberapa variabel yang memiliki Positive Correlations dan Negative Correlations
   

## Data Preparation
Setelah menilai dari data tahap Data Understanding. Maka beberapa tahapan Data Preparation sebagai berikut: 
1. Stemming Kalimat Desc, filter Stopwords dan Menggabungkan dalam variavbel 'Tags'
   
   Stemming Kalimat Deskripsi, Filter Stopwords, dan Penggabungan ke dalam Variabel 'Tags'
Pada dataset tourism_with_id.csv, terdapat kolom deskripsi yang berisi penjelasan mengenai masing-masing destinasi wisata. Teks pada kolom ini diproses dengan melakukan stemming untuk mengubah kata ke bentuk dasarnya dan filtering stopwords untuk menghilangkan kata-kata umum yang tidak memiliki makna signifikan (misalnya: "yang", "dan", "di", dll). Setelah tahap tersebut, kolom deskripsi digabungkan dengan kolom-kolom lain yang relevan seperti category ke dalam satu kolom baru bernama tags.
    > TF-IDF menjadi dasar perhitungan *cosine similarity* pada model content-based filtering, sehingga penting untuk         mengidentifikasi fitur-fitur unik dari setiap tempat wisata.
    
2. TF-IDF Vectorizer
   
   Setelah kolom tags terbentuk, dilakukan proses TF-IDF Vectorization untuk mengubah data teks menjadi representasi numerik vektor. Proses ini penting untuk mengukur bobot atay kepentingan suatu kata dalam satu destinasi dibandingkan dengan seluruh destinasi lainnya.
   > TF-IDF menjadi dasar perhitungan cosine similarity pada model content-based filtering, segingga untuk mengidentifikasi fitur-fitur unik dari setiap tempat wisata
   
3. Encoding
   
   Untuk model collaborative filtering berbasis deep learning, diperlukan data input dalam bentuk numerik. Oleh karena itu, dilakukan proses **encoding** pada ID pengguna dan ID tempat wisata agar dapat diterima oleh model.
   > Proses ini membantu memetakan nilai kategori (user dan item) ke dalam format numerik yang dapat digunakan dalam embendding layer dari model deep learning
   
4. Splitting Data 80:20
   
   Pada data tourism_rating.csv, data dibagi menjadi 80% data latih dan 20% data uji. pembagian ini dilakukan untuk mengukur peforma model collaborative filtering dengan akurasi yang lebih realistis. Data latih digunakan untuk melatih model, sementara data uji digunakan untuk mengevaluasi model setelah pelatihan.
   > Pembagian data ini penting untuk menghindari overfitting serta untuk menguji generalisasi model terhadap data baru
## Modeling
Proyek ini dilakukan pembuatan sistem rekomendasi untuk memberikan Top-N rekomendasi destinasi wisata kepada pengguna berdasarkan preferensi dan riwayat interaksi mereka. Sistem dibangun menggunakan dua pendekatan utama, yaitu:
### 1. **Content-Based Filtering**
   Pendekatan ini menggunakan informasi dari nama tempat destinasi wisata untuk merekomendasikan tempat-tempat yang mirip dengan yang disuaki ata pernah dinilai tinggi oleh pengguna.
   #### Langkah-Langkah:
   - Data deskripsi tempat wisata dibersihkan dan diproses
   - Membuat kolom tags sebagai gabungan fitur deskripsi dan kategori
   - Mencari kemiripan antar destinasi yang dihitung menggunakan Cosine Similarity
   - Memberikan Top-N rekomendasi destinasi wisata yang paling mirip berdasarkan destinasi yang pernah disukai pengguna
   #### Kelebihan:
   - Mampu memberikan rekomendasi personal meskipun tanpa data pengguna lain
   - Mudah diterapkan dan tidak memrlukan bannyak data historus pengguna
   #### Kekurangan:
   - Tidak bisa merekomendasikan tempat baru jika belum pernah ada interaksi
   - Rekomendasi terbatas pada kemiripan fitur, tidak mempertimbangkan preferensi kolektif pengguna lain
#### Output
![Screenshot 4](img/cbf.png)

### 2. **Collbaoritve Filtering (Deep Learning)**
Pendekatan ini memanfaatkan interaksi pengguna dan tempat wisata dalam bentuk rating, menggunakan pendekataan pembelajaran mendalam (deep learning) untuk mempelajari hubungan laten antara pengguna dan item
  #### Langkah-Langkah:
  **Arsitektur Model**
  - User Embedding Layer: Mengubah setiap user_id menjadi vektor berdimendsi 50 yang memrepresentasikan preferensi laten pengguna
  - Place Embedding Layer: Mengubah setiap place_id menjadi vektor berdimensi 50 yang merepresentasikan karakteristik laten dari tempat wisata
  - User dan Place Bias: Tambahan bias untuk menyesuaikan prediksi nilai rating yang lebih realistis
  - Dot Product: Menghitung kesamaan antara user dan place dengan menggunakan operasi dot product antar embedding
    **Proses Pelatihan**
    ![Screenshot 6](img/arsitektur%20model.png)
    ![Screenshot 7](img/model%20compile.png)

    Model di compile dengan fungsi loss Binary Crossentropy, menggunakan optimizer Adam dengan learning rate 0.001, serta metrik evaluasi Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi
   #### Kelebihan:
   - Mampu menangkap relasi kompleks
   -  Fleksibel dan dapat dikembangkan lebih lanjut
   -  Efisien untuk dataset besar
   #### Kekurangan:
   - Masalah Cold Start
   - Memerlukan banyak data pelatihan
   - Interpretasi model sulit

   #### Output
   ![Screenshot 5](img/collaboraitve.png)

## Evaluation
Untuk mengevaluasi peforma sistem rekomendasi yang dibangun, metrik yang digunakan adalah Root Mean Squared Error (RMSE). Pemilihan metrik ini sesuai dengan karakteristik data dan tujuan model, yaitu untuk mengukur seberapa dekat prediksi rating sistem terhadap rating aktual dari pengguna

### Alasan Pemilihan RMSE
RMSE dipilih karena: 
- Cocok untuk masalah regresei seperti prediksi rating dalam sistem rekomendasi 
- Memberikan penalti yang lebih besar terhadap kesalahan predisksi yang besar, sehingga mendorong model untuk lebih akurat pada semua prediksi
- Banyak digunakan dalam evaluasi sistem collaborative filtering, sehingga memudahkan perbandingan antar pendekatan

### Rumus RMSE

RMSE dihitung dengan rumus berikut:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 }
$$

Di mana:
- $\hat{y}_i$ = nilai prediksi (predicted rating)
- $y_i$ = nilai aktual (actual rating)
- $n$ = jumlah total data

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

### Hasil Evaluasi
![Screenshot RMSE](img/rmse.png)
> **Catatan**: Untuk pendekatan content-based filtering, karena sistem ini tidak menghasilkan prediksi rating, maka evaluasi dilakuakn secara kualitatif, misalnya dengan melihat relevansi Top-N rekomendasi yang dihasilkan

### Interpretasi Nilai RMSE
- Nilai RMSE sebesar 0.36 mengindikasikan prediksi rating yang diberikan oleh model rata-rata meleset sebesar 0.36 dari rating sebenarnya dalam skala [0, 1]
- Nilai ini menunjukkan bahwa model cukup akurat dalam memahami preferensi pengguna terhadap tempat wisata, dan cukup baik dalam melakukan generalisasi terhadap data uji. 


