# O‘zbekcha hikoyalar — LLM Midterm Dataset Builder

**Maqsad:** LLM uchun xom hikoya matnlaridan sifat nazoratidan o‘tgan, PII yashirilgan va token IDlarga aylantirilgan `train.bin`/`val.bin` tayyorlash. Ushbu loyiha **modelni o‘qitmaydi**; Dataset Builderda tugaydi.

**Til:** barcha kod izohlari, bosqichlar va hujjatlar o‘zbek tilida.

## Midterm talablariga mos bosqichlar

| Talab | Fayl | Amal |
|---|---|---|
| 0. Repo + dataset olish | `src/fetch.py`, `data/source/hikoyalar_manba.jsonl` | Clean clone dan oflayn manba nusxasi |
| 1. Cleaning | `src/cleaning.py` | Unicode NFC, tutuq belgisi va bo‘shliqlar |
| 2. Filtering | `src/filtering.py` | Til, uzunlik, reklama/URL, harflar ulushi |
| 3. Deduplication | `src/deduplication.py` | SHA256 bilan aniq, word-shingle Jaccard bilan yaqin dublikat |
| 4. PII Scrubbing | `src/pii.py` | Email va telefon namunalarini yashirish |
| 5. Stats Report | `src/stats.py` | Hikoya, so‘z, belgi, filtr, dublikat va PII hisobotlari |
| 6. Tokenizer | `src/tokenizer_bpe.py` | UTF-8 Byte-Level BPE, faqat TRAINda o‘rganish |
| 7. Dataset Builder | `src/dataset_builder.py` | `train.bin`, `val.bin`, `meta.json` |
| Umumiy ishga tushirish | `src/pipeline.py`, `scripts/run_all.py` | Barcha bosqichlarni ketma-ket bajarish |
| Sinovlar | `tests/test_pipeline.py` | 6+ unit/integratsion test |
| Notebook | `notebooks/01_hikoya_pipeline.ipynb` | Bosqichlar, kod izohi, natija va grafik |
| Dataset manbasi/litsenziyasi | **`DATASET_README.md`** | Kelib chiqishi, litsenziya shartlari, cheklovlar |

## Loyiha tuzilishi

```text
uzbek_hikoyalar_midterm/
├── configs/data.yaml
├── data/
│   ├── source/hikoyalar_manba.jsonl   # repo ichidagi boshlang‘ich hikoyalar
│   ├── raw/hikoyalar_raw.jsonl        # xom nusxa
│   ├── processed/                    # cleaning, filtering, dedup, pii
│   ├── splits/train.jsonl, val.jsonl
│   └── binaries/train.bin, val.bin, meta.json
├── src/                               # bosqichma-bosqich Python kodlari
├── scripts/run_all.py
├── scripts/check_project.py
├── tests/test_pipeline.py
├── tokenizer/tokenizer.json
├── notebooks/01_hikoya_pipeline.ipynb
├── results/stats_report.json, stats_report.md
├── DATASET_README.md
└── requirements.txt
```

Loyiha tuzilishi
uzbek_hikoyalar_midterm/
│
├── .venv/                 # Python virtual muhiti
├── configs/               # Konfiguratsiyalar
├── Data/                  # Dataset
├── src/                   # Pipeline kodlari
├── scripts/               # Ishga tushirish kodlari
├── notebooks/             # Jupyter Notebook
├── tests/                 # Testlar
├── results/               # Natijalar
│
├── requirements.txt
├── README.md
├── DATASET_README.md
└── .gitignore

## 1. Noldan ishga tushirish (Windows 11 / VS Code)

Loyihani arxivdan chiqarib, uning bosh papkasida terminal oching:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py
python scripts/check_project.py
python -m unittest discover -s tests -v
```

Linux / macOS da faollashtirish: `source .venv/bin/activate`. VS Code Jupyter notebook uchun o‘sha `.venv` interpreterini tanlang; agar Jupyter paketlari yo‘q bo‘lsa `pip install jupyterlab ipykernel` bajaring. Notebookni `notebooks/01_hikoya_pipeline.ipynb` dan ochib, **Run All** qiling. U loyiha asosiy papkasini o‘zi topadi. Internet faqat paketlar o‘rnatishda kerak bo‘lishi mumkin; ma’lumotlar repo ichida.

## 2. Foydalanish

- Datasetni sozlash: `configs/data.yaml`.
- Bitta buyruqda yig‘ish: `python scripts/run_all.py`.
- Barcha qadamlarni ko‘rib o‘rganish: `notebooks/01_hikoya_pipeline.ipynb`.
- Matnni tekshirish: `data/processed/04_pii_safe.jsonl`.
- Statistikani ko‘rish: `results/stats_report.md`.
- Keyingi GPT training uchun: `data/binaries/train.bin`, `val.bin`, `meta.json`, `tokenizer/tokenizer.json`.

### Nega manba internetdan yuklanmaydi?

Bu Midterm uchun ataylab yozilgan **original o‘quv hikoyalari** ishlatildi, TinyStories yoki uchinchi tomon kitoblaridan nusxa olinmadi. Manba `data/source` katalogida berilgan. `src/fetch.py` uni `data/raw` ga ko‘chiradi; bu bosqich yuklab olingan manbani alohida xom faylda saqlash tartibini namoyish qiladi, lekin internetdan scraping qilmaydi. Tashqi manba qo‘shishdan oldin ma’lumot litsenziyasini tasdiqlash, robots.txt va sayt shartlarini tekshirish zarur. Manbani `DATASET_README.md` da tushuntirdik.

## 3. Muhim texnik tafsilotlar

- Hikoyalar **dedupdan keyin** train/valga ajratiladi (bir xil hikoyaning ikki splitga tushishini kamaytirish).
- BPE tokenizer **faqat training split** asosida o‘rganadi (validationdan statistik sizib kirish bo‘lmasligi uchun).
- Maxsus token `<eos>` IDsi = 256. Dastlabki 256 ID UTF-8 baytlarga ajratilgan.
- `train.bin` va `val.bin` — ketma-ket **uint16 little-endian** token IDlar; har hikoya oxirida `<eos>` bor.
- `tokenizer/tokenizer.json` **maxsus o‘quv formati**. Hugging Face `tokenizers` bilan bevosita mos **emas**; uni `SimpleByteBPE.from_file` bilan oching.
- `block_size`, `batch_size` va GPT arxitekturasi keyingi **Training** bosqichida tanlanadi; bu loyiha Training Loop emas.
- PII qoidalari namuna uchun: regex faqat email/telefonni yashiradi; ism, manzil va real PII uchun kengroq audit kerak.
- 4-gram Jaccard kichik dataset uchun ishlatilgan. Katta datasetda MinHash/LSHni ko‘rib chiqing.
- Korpus kichik va AI yordamida tuzilgan, murakkab til modeli sifatini kafolatlamaydi.

## 4. Midterm uchun tekshirish tartibi

1. `python scripts/run_all.py` xatosiz tugashi.
2. `python scripts/check_project.py` ishlashi.
3. `python -m unittest discover -s tests -v` testlari o‘tishi.
4. Notebook `Run All` ishlashi.
5. `results/stats_report.md` va `data/binaries/meta.json` yaratilishi.
6. `DATASET_README.md` ichida kelib chiqish, litsenziya va cheklovlar yozilganligi.

## 5. Qo‘shimcha tekshiruv va halollik

Rasmda keltirilgan "dataset download" talabi bu yerda oflayn manba nusxasi bilan bajariladi; **vebdan ma’lumot yig‘ish kodi kiritilmagan**, chunki mualliflik huquqi noma’lum matnlarni avtomatik ko‘chirib olish xavfli. "Binaries" uchun tayyor model checkpointi emas, **tokenlashtirilgan dataset binarlari** yaratilgan.

Dataset tafsilotlari: **[DATASET_README.md](DATASET_README.md)**.
