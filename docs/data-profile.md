# Profil dan Audit Dataset Terpilih

## Sumber

- UCI: [Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata)
- Kutipan: Tomczak (2016), DOI [10.24432/C5F600](https://doi.org/10.24432/C5F600)
- Lisensi: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- File: subset `5year.arff` dari paket UCI; salinan kerja ada di `data/raw/polish_companies_5year.arff`.

## Ringkasan terukur

| Pemeriksaan | Hasil |
|---|---:|
| Baris | 5.910 |
| Prediktor | 64 rasio finansial (`Attr1`–`Attr64`) |
| Target | `class` (0/1), dipetakan ke `default_next_year` |
| Event bangkrut | 410 |
| Non-event | 5.500 |
| Event rate sampel | 6,94% |
| Ukuran ARFF terpilih | 2.899.490 byte |
| Missing terbanyak | `Attr37`: 2.548 (43,1%) |
| Baris dengan minimal satu missing | 2.879 (48,7%) |
| Total sel fitur missing | 4.666 (1,23% dari sel fitur) |
| Exact duplicate vector + target | 60 baris; tidak ada label konflik |
| Panel setelah deduplikasi | 5.850 baris: 408 event, 5.442 non-event |
| Tahun/ID perusahaan di file ARFF | Tidak ada kolom eksplisit |

UCI memaparkan bahwa file `5thYear` memakai rasio tahun kelima dan label kebangkrutan setelah satu tahun. Berkas yang diunduh berisi 64 atribut bernama `Attr1`–`Attr64` dan `class`; kami mempertahankan skema tersebut. Metadata laman UCI menyebut field `year`, tetapi field itu tidak muncul dalam ARFF yang kami audit, sehingga tidak digunakan.

## Missingness dan rentang

Atribut finansial adalah rasio; beberapa memiliki distribusi sangat miring atau outlier ekstrem. Audit nilai dan ringkasan missingness dihitung dari data kerja lokal; tabel CSV hasil audit tidak disertakan dalam repositori.

Missing terbanyak ada pada `Attr37`, `(current assets − inventory) / long-term liabilities`. Missingness berbeda menurut target: `Attr37` hilang pada 51% baris bangkrut vs 43% non-bangkrut; `Attr27` pada 30% vs 5%; `Attr21` pada 24% vs kurang dari 0,1%. Ini sinyal asosiasi, bukan bukti sebab-akibat; telusuri provenance sebelum memutuskan apakah indikator missing layak dipakai.

Dari 5.910 baris, hanya 3.031 yang lengkap. Baris lengkap mencakup 102 dari 410 event bangkrut; jadi complete-case menghapus 308 event (75,1%) dan menurunkan event rate menjadi 3,37%. Complete-case hanya menjadi analisis sensitivitas, bukan pipeline utama.

Kebijakan awal:

- ubah `?` menjadi missing, bukan nol;
- jadikan imputasi plus indikator missing sebagai kandidat utama; pelajari parameter hanya dari fold training;
- pelajari batas outlier/winsorization hanya dari fold training;
- lakukan sensitivitas dengan dan tanpa atribut yang missing-nya tinggi;
- deduplicate 60 exact duplicate vector + target sebelum split untuk menghindari kebocoran antar fold;
- simpan tabel perubahan data dan alasan setiap perlakuan.

## Definisi target dan batas validitas

Target berasal langsung dari `class`; tidak perlu menebak event dari perusahaan yang menghilang. File `5year` merupakan task satu tahun ke depan menurut dokumentasi UCI. Event rate 6,94% adalah proporsi file benchmark; jangan menganggapnya sebagai prevalensi portofolio riil atau memakai probabilitas terkalibrasi sebagai PD populasi tanpa koreksi prior dan validasi eksternal.

Dataset tidak membawa pengenal perusahaan atau tahun observasi dalam berkas ARFF. Kita tidak bisa memastikan pemisahan perusahaan yang sama antar fold, membuat group split per entitas atau temporal holdout, atau mengukur perubahan rezim makro. Batasan ini harus dinyatakan dalam README dan memo risiko.

## Pemeriksaan target dan struktur

Konversi awal menghasilkan 5.910 baris, 410 target positif, 5.500 target negatif, dan 4.666 sel fitur missing. Semua target adalah 0/1. Deduplikasi deterministik berdasarkan 64 fitur + target menghapus 60 salinan identik (58 non-event dan 2 event), menghasilkan panel 5.850 baris: 408 event dan 5.442 non-event. File awal tetap tersedia untuk audit; panel deduplikasi menjadi input modeling.
