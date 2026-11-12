# Kamus 64 Rasio Finansial

Kolom ARFF menggunakan nama `Attr1`–`Attr64`; halaman UCI memaparkan formula berurutan sebagai X1–X64. Mapping di bawah mempertahankan urutan tersebut. Formula berasal dari [UCI dataset page](https://archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata).

| Kolom | Formula/definisi sumber | Kelompok analisis |
|---|---|---|
| Attr1 | net profit / total assets | Profitabilitas |
| Attr2 | total liabilities / total assets | Leverage |
| Attr3 | working capital / total assets | Likuiditas |
| Attr4 | current assets / short-term liabilities | Likuiditas |
| Attr5 | [(cash + short-term securities + receivables − short-term liabilities) / (operating expenses − depreciation)] × 365 | Likuiditas/operasi |
| Attr6 | retained earnings / total assets | Retained earnings |
| Attr7 | EBIT / total assets | Profitabilitas |
| Attr8 | book value of equity / total liabilities | Solvabilitas |
| Attr9 | sales / total assets | Efisiensi aset |
| Attr10 | equity / total assets | Struktur modal |
| Attr11 | (gross profit + extraordinary items + financial expenses) / total assets | Profitabilitas |
| Attr12 | gross profit / short-term liabilities | Likuiditas/profitabilitas |
| Attr13 | (gross profit + depreciation) / sales | Margin |
| Attr14 | (gross profit + interest) / total assets | Profitabilitas |
| Attr15 | (total liabilities × 365) / (gross profit + depreciation) | Beban utang |
| Attr16 | (gross profit + depreciation) / total liabilities | Solvabilitas |
| Attr17 | total assets / total liabilities | Solvabilitas |
| Attr18 | gross profit / total assets | Profitabilitas aset |
| Attr19 | gross profit / sales | Margin |
| Attr20 | (inventory × 365) / sales | Perputaran persediaan |
| Attr21 | sales in year n / sales in year n−1 | Pertumbuhan |
| Attr22 | profit on operating activities / total assets | Profitabilitas operasi |
| Attr23 | net profit / sales | Margin bersih |
| Attr24 | gross profit (in 3 years) / total assets | Tren profitabilitas |
| Attr25 | (equity − share capital) / total assets | Struktur modal |
| Attr26 | (net profit + depreciation) / total liabilities | Cakupan utang |
| Attr27 | profit on operating activities / financial expenses | Interest coverage |
| Attr28 | working capital / fixed assets | Likuiditas |
| Attr29 | logarithm of total assets | Ukuran perusahaan |
| Attr30 | (total liabilities − cash) / sales | Leverage bersih |
| Attr31 | (gross profit + interest) / sales | Margin |
| Attr32 | (current liabilities × 365) / cost of products sold | Siklus utang dagang |
| Attr33 | operating expenses / short-term liabilities | Beban operasi/likuiditas |
| Attr34 | operating expenses / total liabilities | Beban operasi/utang |
| Attr35 | profit on sales / total assets | Profitabilitas aset |
| Attr36 | total sales / total assets | Efisiensi aset |
| Attr37 | (current assets − inventories) / long-term liabilities | Likuiditas |
| Attr38 | constant capital / total assets | Struktur modal |
| Attr39 | profit on sales / sales | Margin |
| Attr40 | (current assets − inventory − receivables) / short-term liabilities | Likuiditas |
| Attr41 | total liabilities / ((profit on operating activities + depreciation) × (12/365)) | Cakupan utang |
| Attr42 | profit on operating activities / sales | Margin operasi |
| Attr43 | receivables turnover + inventory turnover in days | Siklus konversi kas |
| Attr44 | (receivables × 365) / sales | Hari piutang |
| Attr45 | net profit / inventory | Profitabilitas/persediaan |
| Attr46 | (current assets − inventory) / short-term liabilities | Likuiditas |
| Attr47 | (inventory × 365) / cost of products sold | Hari persediaan |
| Attr48 | EBITDA (profit on operating activities − depreciation) / total assets | Profitabilitas aset |
| Attr49 | EBITDA (profit on operating activities − depreciation) / sales | Margin EBITDA |
| Attr50 | current assets / total liabilities | Likuiditas/solvabilitas |
| Attr51 | short-term liabilities / total assets | Leverage jangka pendek |
| Attr52 | (short-term liabilities × 365) / cost of products sold | Siklus utang dagang |
| Attr53 | equity / fixed assets | Struktur modal |
| Attr54 | constant capital / fixed assets | Struktur modal |
| Attr55 | working capital | Besaran likuiditas |
| Attr56 | (sales − cost of products sold) / sales | Margin kotor |
| Attr57 | (current assets − inventory − short-term liabilities) / (sales − gross profit − depreciation) | Likuiditas/operasi |
| Attr58 | total costs / total sales | Beban biaya |
| Attr59 | long-term liabilities / equity | Leverage jangka panjang |
| Attr60 | sales / inventory | Perputaran persediaan |
| Attr61 | sales / receivables | Perputaran piutang |
| Attr62 | (short-term liabilities × 365) / sales | Siklus utang dagang |
| Attr63 | sales / short-term liabilities | Likuiditas |
| Attr64 | sales / fixed assets | Efisiensi aset tetap |

Sebagian rasio memiliki denominator yang dapat nol atau mendekati nol dan menghasilkan angka ekstrem. Kelompok di atas membantu interpretasi, bukan hasil seleksi final. Pemeriksaan missingness dan rentang dilakukan pada data kerja; tabel hasil CSV tidak disertakan dalam repositori.
