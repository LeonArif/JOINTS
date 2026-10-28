# Konten PPT JOINTS: dari Blend sampai Closing

Sumber angka: `leon/dikeri_ivant.ipynb` (output yang tersimpan) dan `leon/logs.md`.
Semua angka di bawah disalin dari sumber itu, tidak ada yang dikarang.
Tanda `[HITUNG]` berarti angkanya belum ada di output notebook dan harus dijalankan dulu.
Tanda `[KEPUTUSAN]` berarti ada hal yang harus kamu putuskan sebelum slide dibuat.
Tag slide: B = bisnis, T = teknis, X = jembatan (judul bisnis, bukti teknis).

## Keputusan yang perlu kamu ambil dulu

1. **Blend tidak mengalahkan Kumo di CV.**
   CV blend 0.3251 dan Kumo sendiri 0.3250, jadi di CV keduanya seri.
   Draft kamu menulis "blend mengalahkan masing-masing model", dan itu tidak benar untuk Kumo.
   Konten di bawah memakai klaim yang benar: blend mengalahkan LGBM dan NN, seri dengan Kumo, dan tidak ada satu model pun yang terbaik di semua fold.
2. **Bobot 0.4 / 0.3 / 0.3 bukan hasil pencarian terbaik di CV.**
   Di CV, Kumo 0.7 / LGBM 0.3 memberi 0.3239, lebih baik dari 0.3251.
   Bobot 0.4 / 0.3 / 0.3 dipertahankan karena lift postprocessing ditemukan di basis itu.
   Slide Blend menulis ini apa adanya (selisih antar bobot hanya di angka keempat).
3. **Klaim "NN membantu" hanya terbukti di leaderboard publik, tidak di privat.**
   Jangan dipakai sebagai bukti di slide kecuali kamu menyebut sumbernya.
4. **Lift D7-D10 x1.2 dan Ramadan x1.1 dikalibrasi dari leaderboard publik dan merugikan di privat.**
   File 0.38002: publik 0.38002, privat 0.39982. File tanpa lift: publik 0.38432, privat 0.39763.
   Pilihan: (a) tetap tampil di slide Postprocessing dan diakui di Saran sebagai keterbatasan (disarankan, juri teknis akan memeriksa), atau (b) dibuang dari slide.
5. **Postprocessing: ikut atau tidak?** Disarankan ikut, satu slide saja, karena 23,9% baris test (17.327 dari 72.611) jatuh di Ramadan atau Lebaran yang tidak ada di train.
6. **Urutan slide.**
   Di draft kamu KESIMPULAN ada sebelum XAI 3.
   Konten ini menaruh XAI 1, 2, 3 dulu, baru KESIMPULAN, supaya kesimpulan tidak memuat hal baru.
7. **Angka baseline naif 0.7393** dihitung di notebook awal pada 56.049 baris, sedangkan model akhir dihitung pada 52.905 baris (setelah hari tidak lengkap dibuang).
   Disarankan dihitung ulang pada 52.905 baris supaya sebanding `[HITUNG]`.
8. **Slide fase (opsional)** tidak dibuat.
   Slide ringkasan Fase 4 bisa ditambah bila batas slide masih longgar.

## Ledger angka

| Angka | Nilai | Sumber |
|---|---|---|
| Periode train | 2025-04-01 sampai 2025-09-30 | notebook sel 13 |
| Periode test | 2025-10-04 sampai 2026-03-27 | notebook sel 13 |
| Baris train / baris tambahan / baris test | 52.905 / 244.229 / 72.611 | notebook sel 64 |
| Jumlah fitur | 77 | notebook sel 64 |
| CV MASE Kumo | 0.3250 | notebook sel 70 |
| CV MASE LGBM | 0.3301 | notebook sel 66 |
| CV MASE NN | 0.3494 | notebook sel 72 |
| CV MASE Blend (0.4 / 0.3 / 0.3) | 0.3251 | notebook sel 76 |
| Naif: rata-rata D1-D3 | 0.7393 | logs.md 2026-09-28 |
| Naif: tiket D3 | 0.6965 | logs.md 2026-09-28 |
| Naif: prediksi 0 | 0.5877 | logs.md 2026-09-28 |
| Penurunan blend vs naif 0.7393 | 56,0% | dihitung: (0.7393 - 0.3251) / 0.7393 |
| Penurunan blend vs naif 0 | 44,7% | dihitung: (0.5877 - 0.3251) / 0.5877 |
| Leaderboard file notebook ini | publik 0.38002, privat 0.39982 | logs.md 2026-10-19 |
| Leaderboard file tanpa lift | publik 0.38432, privat 0.39763 | logs.md 2026-10-19 |

CV per fold (sel 66, 70, 72, 74):

| Model | CV | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| Kumo | 0.3250 | 0.3201 | 0.3613 | 0.2025 | 0.3311 | 0.4101 |
| LGBM | 0.3301 | 0.3313 | 0.3533 | 0.2117 | 0.3465 | 0.4081 |
| NN | 0.3494 | 0.3671 | 0.3850 | 0.2345 | 0.3602 | 0.4004 |
| Blend | 0.3251 | `[HITUNG]` | `[HITUNG]` | `[HITUNG]` | `[HITUNG]` | `[HITUNG]` |

Blend per horizon (sel 76): D4 0.3424, D5 0.3569, D6 0.3260, D7 0.3228, D8 0.2907, D9 0.3147, D10 0.3219.

---

## SLIDE: Blend (T)

**Judul:** Blend menggabungkan tiga model dengan bobot Kumo 0.4, LGBM 0.3, dan NN 0.3

**Diagram (alur kiri ke kanan):**
Kumo-Tabular (bobot 0.4), LightGBM (bobot 0.3), dan Neural Network (bobot 0.3) masing-masing memprediksi target_ratio.
Ketiganya masuk ke rata-rata berbobot.
Hasilnya dipotong: nilai di bawah 0.1 menjadi tepat 0.
Prediksi akhir dikali scale menjadi jumlah tiket D4-D10.

**Bagaimana digabung:**
- Ketiga model memprediksi target yang sama (target_ratio) pada baris yang sama, sehingga angkanya bisa dirata-ratakan langsung.
- Prediksi blend = 0.4 x Kumo + 0.3 x LGBM + 0.3 x NN.
- LGBM dan NN sudah memotong prediksi di bawah 0.2 menjadi 0 sebelum digabung.
  Kumo tidak dipotong karena sebagai median dari kuantil ia sudah menghasilkan 0 sendiri.
- Setelah digabung, nilai di bawah 0.1 dipotong lagi menjadi 0, dengan alasan yang sama seperti zero-snap per model.

**Bagaimana bobot dipilih:**
- Kumo mendapat bobot terbesar karena skor CV-nya paling baik (0.3250).
- LGBM dan NN diberi bobot sama karena kesalahan mereka berbeda dari Kumo, sehingga keduanya tetap ikut menyumbang.
- Bobot ditetapkan tetap dan bukan hasil pencarian otomatis, karena beda skor antar kombinasi bobot sangat kecil (tabel di bawah).

**Tabel sensitivitas bobot (CV MASE, sumber logs.md 2026-10-11):**

| Kombinasi | CV MASE |
|---|---|
| Kumo 0.7 / LGBM 0.3 | 0.3239 |
| Kumo 0.8 / LGBM 0.2 | 0.3239 |
| Kumo 0.6 / LGBM 0.4 | 0.3242 |
| Kumo 0.5 / LGBM 0.5 | 0.3249 |
| Kumo sendiri | 0.3250 |
| **Kumo 0.4 / LGBM 0.3 / NN 0.3 (dipakai)** | **0.3251** |

**Artinya bagi bisnis:** hasilnya tidak bergantung pada satu model, dan perubahan bobot kecil tidak mengubah kualitas prediksi.

**Catatan bicara:**
- Jujur sebut bahwa kombinasi tanpa NN sedikit lebih baik di CV (selisih 0.0012).
- Alasan NN tetap ada: kesalahannya berbeda, dan basis ini dipakai saat kalibrasi postprocessing.
- `[KEPUTUSAN]` nomor 2 dan 3 di atas.

---

## SLIDE: Mengapa blend bekerja (X)

**Judul:** Tidak ada satu model yang terbaik di semua fold, sehingga blend menjaga skor setara model terbaik

**Metode (satu baris):** Skor setiap model dihitung pada 5 fold yang dikelompokkan per film, lalu blend dihitung dari prediksi out-of-fold yang sama.

**Bukti 1: model terbaik berganti-ganti (data sudah ada).**
Tabel per fold di ledger. Sel terbaik per fold ditebalkan:
- Fold 0: Kumo (0.3201).
- Fold 1: LGBM (0.3533).
- Fold 2: Kumo (0.2025).
- Fold 3: Kumo (0.3311).
- Fold 4: NN (0.4004).

Berdasarkan tabel bisa dilihat Kumo terbaik di 3 fold, LGBM di 1 fold, dan NN di 1 fold.
Kolom Blend per fold ditambahkan setelah dihitung `[HITUNG]`.

**Bukti 2: ketiga model mengikuti kurva peluruhan yang sama (`[HITUNG]`).**
Grafik garis: median target_ratio aktual per horizon D4-D10 dibandingkan prediksi out-of-fold Kumo, LGBM, NN, dan Blend.
Dari output yang sudah ada, kurva aktual turun dari 0.915 (D4) ke 0 (D9 dan D10), dan error blend per horizon stabil di 0.2907 sampai 0.3569.

**Bukti 3: kesalahan ketiga model tidak sama (`[HITUNG]`).**
Korelasi error out-of-fold antar model (tiga pasangan) dan skor Blend dibandingkan skor tiap model pada baris yang sama.
Hanya klaim "kesalahan berbeda" yang boleh ditulis bila korelasinya memang rendah. Bila tinggi, ganti kalimatnya menjadi "blend tidak merugikan".

**Hasil yang boleh diklaim (CV):**
- Blend 0.3251 lebih baik dari LGBM (0.3301) dan NN (0.3494).
- Blend seri dengan Kumo (0.3250), selisih 0.0001.

**Artinya bagi bisnis:** model terbaik bergantung pada film yang masuk, dan film test belum pernah dilihat, jadi blend memilih jalan yang paling aman untuk film baru.

**Catatan bicara:** sebut sumber angka (CV out-of-fold, bukan leaderboard) di judul gambar atau catatan kaki.

---

## SLIDE: Mengapa kita memilihnya untuk dataset (X)

**Judul:** Setiap kelebihan model menjawab satu temuan EDA, dan setiap kekurangan sudah ditangani

**Metode (satu baris):** Kebutuhan dari data dicocokkan satu per satu dengan kelebihan model, lalu kekurangannya dicatat bersama penanganannya.

Kolom kebutuhan harus disamakan dengan nomor temuan EDA di deck kamu. Isi di bawah memakai temuan dari notebook dan log.

| Kebutuhan dari data (EDA) | Model dan kelebihan yang menjawab | Kekurangan | Penanganan |
|---|---|---|---|
| Target banyak yang tepat 0 (median target_ratio 0 di D9 dan D10) | Kumo menghasilkan median dari kuantil, sehingga bisa tepat 0 | LGBM kurang tajam di nol dan nilai ekstrem | Prediksi di bawah 0.2 dijadikan 0, blend di bawah 0.1 dijadikan 0 |
| Film test belum pernah dilihat dan hanya ada 205 film train | Kumo pretrained, tidak perlu belajar dari nol pada 205 film | Baris test harus mirip baris konteks | Konteks 9.000 baris asli + 3.000 baris geser, 3 seed konteks dirata-ratakan, validasi per film |
| Akhir pekan, libur, dan long weekend mengubah penjualan | LGBM membelah pada hari dan blok libur (dow, off_block_len, tgt_off) | Tree tidak bisa memprediksi pola yang belum pernah dilihat | Postprocessing musim untuk Ramadan dan Lebaran |
| 121 bioskop dan 69 kota berperilaku berbeda | NN dengan embedding, bioskop dan kategori mirip saling berdekatan | NN mudah overfit dan butuh penyiapan manual | Input dipotong di persentil 1 dan 99 lalu distandarisasi, dropout 0.287, weight decay, 5 seed, baris geser berbobot 0.5 |
| Skala bioskop besar dan kecil berbeda jauh | Semua model memprediksi target_ratio dengan loss absolut | Tidak ada | Loss L1 pada target_ratio sama dengan metrik MASE |
| Kumo berat dan butuh GPU | Tidak ada | Lambat | Konteks dibatasi 12.000 baris dan bobot Kumo diunduh, tidak disimpan dalam model ekspor |

**Artinya bagi bisnis:** pilihan model mengikuti sifat data bioskop (banyak nol, film baru, kalender), bukan mengikuti model yang sedang populer.

**Catatan bicara:** tutup dengan "Itulah sebabnya ketiganya dipakai bersama dan tidak dipilih salah satu".

---

## SLIDE: Postprocessing (X) `[KEPUTUSAN]` nomor 4 dan 5

**Judul:** 24% baris test jatuh di Ramadan dan Lebaran yang tidak pernah dilihat model, sehingga levelnya dikoreksi di luar model

**Metode (satu baris):** Model dilatih April sampai September 2025, jadi tiga efek musim test ditangani sesudah prediksi, dengan data nasional musim lalu yang terbit sampai 2025-09-30.

**Mengapa perlu:**
- Test berjalan 4 Oktober 2025 sampai 27 Maret 2026, dan Ramadan jatuh pada 18 Februari sampai 19 Maret 2026.
- Tiga kelompok baris menyentuh Ramadan: ram_to_lebaran 4.598 baris, ram_to_ram 10.767 baris, pre_to_ram 1.962 baris, total 17.327 dari 72.611 baris (23,9%).
- Bukti dari musim lalu: film yang rilis saat Idulfitri 2025 mempertahankan pangsa pasar nasionalnya (rasio pangsa D4-D10 terhadap D1-D3 sekitar 1,0 sampai 1,1), sedangkan film biasa turun sampai 0,18 pada D10 (0,77 pada D4).

**Tiga penyesuaian:**

| Penyesuaian | Isi | Alasan |
|---|---|---|
| 1. Level musim | Level tiap kelompok Ramadan memakai rumus dari chart nasional musim lalu: perubahan pasar nasional pada hari yang sama relatif terhadap Idulfitri, dikali perubahan pangsa film | Model belum pernah melihat Ramadan, dan mean prediksi ram_to_lebaran hanya 0.533 sedangkan rumus memberi 2.205 (faktor 4,857) |
| 2. Faktor genre Ramadan | action 1,27, drama 1,37, family 1,48, horor 0,80, lalu dinormalkan supaya rata-rata kelompok tetap | Pada Ramadan lalu, genre tidak bertahan sama, dan film Ramadan test sebagian besar bukan horor |
| 3. Lift horizon akhir | Horizon D7-D10 dikali 1,2 dan kelompok Ramadan dikali 1,1 | Model dilatih pada musim yang menurun lebih cepat, sehingga hari akhir terlalu rendah |

Tabel faktor level (sel 79):

| Kelompok | Baris | Mean model | Mean rumus | Faktor |
|---|---|---|---|---|
| ram_to_lebaran | 4.598 | 0.533 | 2.205 | 4,857 |
| ram_to_ram | 10.767 | 0.347 | 0.422 | 1,215 |
| pre_to_ram | 1.962 | 0.165 | 0.153 | 0,929 |

**Sumber data (untuk sitasi):** chart nasional harian (Cinepoint) sampai 2025-09-30, kalender cuti bersama dan libur sekolah, dan tabel Ramadan.
Notebook memeriksa bahwa tanggal chart terakhir tidak melewati 2025-09-30.

**Catatan jujur untuk slide atau catatan bicara:**
- Penyesuaian 3 dikalibrasi dari umpan balik submission publik.
  Di leaderboard privat, file dengan lift (0.39982) lebih buruk dari file tanpa lift (0.39763).
- Kalibrasi penyesuaian 1 dan 2 memakai data musim lalu saja, tidak memakai label test.

**Artinya bagi bisnis:** untuk minggu Ramadan dan Lebaran, perkiraan permintaan tidak lagi mengikuti pola turun biasa, sehingga kebutuhan layar dan pertunjukan tidak diremehkan.

---

## SLIDE: Hasil dibandingkan model lain (T)

**Judul:** Blend menurunkan error 56% dari baseline naif 0.7393, dan setara dengan model tunggal terbaik

**Metode (satu baris):** Semua skor adalah MASE out-of-fold pada 5 fold yang dikelompokkan per film, semakin kecil semakin baik.

**Tabel hasil (CV MASE):**

| Model | CV MASE | Turun dari naif 0.7393 |
|---|---|---|
| Naif: rata-rata D1-D3 | 0.7393 | 0% |
| Naif: tiket D3 | 0.6965 | 5,8% |
| Naif: prediksi 0 | 0.5877 | 20,5% |
| Neural network | 0.3494 | 52,7% |
| LightGBM | 0.3301 | 55,4% |
| Kumo-Tabular | 0.3250 | 56,0% |
| **Blend 0.4 / 0.3 / 0.3** | **0.3251** | **56,0%** |

Persen dihitung dari ledger: (0.7393 - skor) / 0.7393, dibulatkan satu desimal.
Catatan: Kumo 56,04% dan Blend 56,03%, jadi dua-duanya tertulis 56,0%.

**Dua kalimat yang menyertai tabel:**
- Terhadap tebakan naif yang paling kuat (prediksi 0, 0.5877), blend menurunkan error 44,7%.
- CV berbeda dari leaderboard (publik 0.38002, privat 0.39982) karena test berisi Ramadan dan Lebaran yang tidak ada di train dan pasarnya lebih sepi `[KEPUTUSAN]` nomor 4.

**Artinya bagi bisnis:** perkiraan permintaan harian menjadi jauh lebih dekat ke kenyataan dibanding hanya melanjutkan angka tiga hari pertama.

**Catatan bicara:** tulis sumber angka (CV out-of-fold) di bawah tabel. Hitung ulang baris naif pada 52.905 baris `[HITUNG]`.

---

## SLIDE: XAI 1, kepentingan fitur global (X)

**Judul:** Ketiga model bertumpu pada fitur yang berbeda, dan itu memperkuat alasan memakai blend `[HITUNG]`

**Metode (satu baris):** Permutation importance dengan cara yang sama untuk ketiga model: satu kolom diacak pada data validasi, lalu dilihat kenaikan MASE, dengan fold yang dikelompokkan per film.

**Yang disiapkan:**
- Permutation importance untuk LGBM, Kumo, dan NN pada 10 sampai 15 fitur teratas (Kumo butuh prediksi ulang di GPU untuk tiap kolom, jadi dibatasi).
- Tiga grafik batang berdampingan dengan skala sumbu yang sama.
- Kolom "tindakan yang didukung" di samping tiap fitur (usulan di tabel bawah).
- Pendamping bukti blend: korelasi error antar model dan skor blend dibandingkan skor tiap model pada data yang sama (sama dengan Bukti 3 slide "Mengapa blend bekerja").

**Usulan istilah tindakan per fitur (sesuaikan dengan hasil):**

| Fitur | Arti dalam bahasa bisnis | Tindakan yang didukung |
|---|---|---|
| mv_t3, mv_clusters, mv_trend | Kekuatan dan luas rilis film secara nasional | Alokasi layar pada film yang bertahan |
| tickets_per_show3 | Permintaan per pertunjukan | Tambah atau kurangi jumlah pertunjukan |
| first_active, eff_age, active_days | Bioskop mulai menayangkan terlambat | Jangan samakan dengan film yang tayang dari hari pertama |
| win_off, tgt_off, off_block_len | Hari libur di sekitar jendela dan hari target | Jadwalkan pertunjukan tambahan pada long weekend |
| cl_scale, city_name | Ukuran dan lokasi bioskop | Keputusan per lokasi, bukan satu angka nasional |
| horizon | Hari ke berapa sejak rilis | Waktu menarik film |

**Dasar fitur yang penting (dari ablasi LGBM, logs.md 2026-10-10):** menghapus grup window_offdays menaikkan MASE dari 0.3313 menjadi 0.3395, dan menghapus film_level menaikkannya menjadi 0.3372. Selisih di bawah sekitar 0.003 termasuk noise antar seed.

**Peringatan:**
- Jangan memakai bobot attention Kumo sebagai feature importance.
- Importance bawaan LGBM (gain) tidak sebanding dengan model lain.
- Importance berbeda hanya penjelas, bukan bukti. Buktinya adalah korelasi error dan skor blend.

**Artinya bagi bisnis:** keputusan yang didukung model bertumpu pada tiga hal yang bisa dikendalikan operator: kekuatan film, permintaan per pertunjukan, dan kalender.

---

## SLIDE: XAI 2, analisis baris target nol (X)

**Judul:** Baris dengan tiket tepat nol adalah sumber error terbesar yang terukur, sehingga zero-snap dipakai `[HITUNG]`

**Metode (satu baris):** Baris out-of-fold dipisah menjadi target nol dan target positif, lalu dihitung porsi baris, porsi total error, dan sebaran prediksi LGBM di baris nol.

**Yang harus dihitung:**
- Persen baris dengan target_ratio tepat 0 pada 52.905 baris train. Angka lama di log: 31,8% pada set fitur sebelumnya, jadi hitung ulang.
- Persen total error yang berasal dari baris nol, sebelum dan sesudah zero-snap.
- Histogram prediksi LGBM di baris nol, dengan garis ambang 0.2.
- Persen prediksi LGBM di bawah 0.2 dibandingkan persen prediksi Kumo di bawah 0.02.

**Fakta pendukung yang sudah ada (set fitur lebih lama, logs.md 2026-10-07):**
- Kumo memprediksi hampir 0 (di bawah 0.02) pada 31,4% baris, sedangkan GBDT yang sudah di-tune hanya 9%.
- Memotong prediksi di bawah 0.2 memperbaiki LightGBM dari 0.3727 menjadi 0.3673.
- Pada kurva peluruhan, median target_ratio sudah 0 pada D9 dan D10.

**Artinya bagi bisnis:** ketika permintaan sudah habis, bioskop mencabut film dan tiket tepat nol, sehingga model yang berani memprediksi nol menghindari menyisakan layar untuk film yang sudah selesai.

---

## SLIDE: XAI 3, mengapa median lebih baik daripada rata-rata untuk MASE (X)

**Judul:** MASE dihitung dari selisih absolut, sehingga prediksi terbaik adalah median dan bukan rata-rata

**Metode (satu baris):** Untuk loss absolut, angka tunggal yang membuat total selisih paling kecil adalah median dari nilai yang mungkin.

**Contoh angka (dibuat sederhana):**
Lima pasangan film-bioskop yang mirip punya target_ratio D9: 0, 0, 0, 0,5, dan 3.

| Tebakan | Selisih tiap pasangan | Total error | Rata-rata error |
|---|---|---|---|
| Rata-rata (0,7) | 0,7 + 0,7 + 0,7 + 0,2 + 2,3 | 4,6 | 0,92 |
| 0,5 | 0,5 + 0,5 + 0,5 + 0 + 2,5 | 4,0 | 0,80 |
| Median (0) | 0 + 0 + 0 + 0,5 + 3 | 3,5 | 0,70 |

Berdasarkan tabel bisa dilihat rata-rata ditarik naik oleh satu nilai besar (3), sedangkan median tidak terpengaruh.
Pada metrik kuadrat (RMSE) rata-rata yang menang, tetapi metrik kompetisi ini absolut.

**Dipakai di pipeline:**
- LGBM dan NN dilatih dengan loss L1.
- Kumo mengeluarkan banyak kuantil dan yang diambil adalah mediannya.
- Zero-snap menjaga median yang bernilai 0 tidak berubah menjadi angka kecil positif.

**Artinya bagi bisnis:** untuk film yang separuh lebih pasangannya sudah tidak laku, prediksi yang jujur adalah nol, bukan angka kecil yang membuat layar tetap dialokasikan.

---

## SLIDE: KESIMPULAN (B)

**Judul:** Tiga tujuan di latar belakang terjawab, dan model menurunkan error 56% dari tebakan naif

Satu tujuan satu baris, dengan angka yang disalin dari ledger.
Isi di bawah memakai kata-kata tujuan dari slide latar belakang kamu.

| Tujuan (slide 2) | Hasil | Angka |
|---|---|---|
| 1. Memprediksi tren D4-D10 dari dinamika D1-D3 dan pola film sebelumnya, tidak hanya melanjutkan angka pembukaan | Fitur pasangan, fitur film, dan baris geser dipakai dan divalidasi pada film yang belum pernah dilihat | MASE blend 0.3251 vs naif 0.7393 (turun 56,0%), error per horizon 0.2907 sampai 0.3569 |
| 2. Menangkap pengaruh kalender (akhir pekan dan hari libur) pada prediksi penjualan harian | Flag libur, cuti bersama, libur sekolah, blok libur, dan hari libur di jendela dipakai sebagai fitur, plus postprocessing Ramadan | Menghapus fitur libur di jendela menaikkan MASE LGBM 0.3313 menjadi 0.3395. Ramadan: 17.327 baris test (23,9%) diberi level musim |
| 3. Memprediksi di level pasangan film-bioskop dengan akurasi relatif terhadap skala | Target_ratio = tiket / scale dengan loss L1, sehingga loss sama dengan metrik | Prediksi untuk 10.373 pasangan test dan 72.611 baris, MASE dihitung per pasangan relatif ke scale-nya |

**Tujuan final:** model forecasting cold-start yang memprediksi tiket harian D4-D10 untuk setiap pasangan film-klaster dari tiga hari pertama sudah dibangun.

**Temuan utama (satu baris):** error terbesar ada di baris yang targetnya tepat nol dan di musim yang tidak ada di train.
**Hasil model dibanding baseline (satu baris):** blend 0.3251 vs naif terkuat (prediksi 0) 0.5877 dan vs naif D1-D3 0.7393.

Tidak ada angka atau klaim baru di slide ini.
Perhatian: baris tujuan 3 belum punya bukti per ukuran klaster. Hitung MASE per kelompok ukuran bioskop `[HITUNG]` bila ingin klaim itu dikuatkan.

---

## SLIDE: SARAN DAN PENGEMBANGAN SELANJUTNYA (B)

**Judul:** Model dijalankan tiap hari setelah D3 dan hasilnya dipakai untuk memutuskan layar dan jumlah pertunjukan

**Blok 1: Saran, memasukkan model ke dalam sistem.**
Alur harian (diagram kiri ke kanan):
1. Data penjualan D1-D3 setiap pasangan film-bioskop ditarik setelah hari ketiga rilis.
2. Fitur dibuat dengan pipeline yang sama dengan latihan (kalender, profil film, data pasar).
3. Model menghasilkan tiket harian D4-D10 untuk setiap pasangan.
4. Dashboard memperlihatkan prediksi per bioskop dan per film untuk 7 hari ke depan.
5. Manajer bioskop menyetujui perubahan jadwal.

**Tindakan ketika model memberi sinyal:**

| Sinyal dari model | Tindakan |
|---|---|
| Prediksi naik pada akhir pekan atau hari libur | Tambah pertunjukan atau pindahkan ke layar yang lebih besar pada hari itu |
| Prediksi turun cepat dibanding D3 | Kurangi pertunjukan dan alihkan layar ke film yang bertahan |
| Prediksi mendekati nol | Siapkan penarikan film dan beri layar pada rilis baru |
| Pasangan yang mulai terlambat | Jangan alokasikan berdasarkan pembukaan, karena umurnya lebih muda dari horizonnya |
| Minggu Ramadan dan Lebaran | Gunakan level musim, jangan memakai pola turun biasa |

**Blok 2: Pengembangan selanjutnya.**
- **Uji coba kecil:** jalankan pada beberapa bioskop dan bandingkan keputusan memakai model dengan keputusan biasa.
- **Pantau dan latih ulang:** hitung MASE mingguan, dan latih ulang saat musim baru masuk.
- **Data lebih banyak:** data lebih dari satu tahun supaya Ramadan dan Lebaran bisa dipelajari langsung, bukan lewat rumus.
- **Rentang prediksi:** Kumo mengeluarkan kuantil, sehingga manajer bisa memilih angka yang aman (misalnya kuantil tinggi saat kapasitas murah).
- **Data operasional:** kapasitas kursi, harga, dan promosi bila operator bisa membagikannya.

**Keterbatasan yang jujur:**
- Validasi hanya mencakup April sampai September, dan Ramadan serta Lebaran ditangani lewat rumus dari data musim lalu.
- Lift horizon akhir dikalibrasi dari leaderboard publik dan tidak terbukti di privat `[KEPUTUSAN]` nomor 4.
- Kumo butuh GPU, dan pasangan yang mulai terlambat tetap sulit diprediksi.

---

## SLIDE: CLOSING (B)

**Isi:** Terima kasih.
Tambahkan nama tim dan satu baris kontak bila lomba memintanya.

Slide cadangan (setelah closing): hyperparameter LGBM dan NN, daftar 77 fitur, tabel CV per fold, dan lingkungan serta waktu jalan.
Ini penjelas untuk juri teknis yang ingin memverifikasi.

---

## Checklist sebelum jadi PPT

- [ ] Putuskan `[KEPUTUSAN]` nomor 1 sampai 8.
- [ ] Jalankan perhitungan `[HITUNG]` dari prediksi out-of-fold: blend per fold, korelasi error, kurva median aktual vs prediksi, porsi baris nol dan porsi error, baris naif pada 52.905 baris, MASE per ukuran klaster, permutation importance.
- [ ] Samakan kolom "kebutuhan dari data" dengan nomor temuan EDA yang sudah kamu tulis.
- [ ] Pastikan tiap angka di slide sama dengan ledger.
- [ ] Setiap grafik menyebut sumber (CV out-of-fold atau leaderboard).
