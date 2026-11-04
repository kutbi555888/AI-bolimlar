# Dataset README — O‘zbekcha qisqa hikoyalar (TinyStories emas)

## 1. Dataset haqida

- **Nomi:** O‘zbekcha qisqa hikoyalar — o‘quv korpusi, 1-versiya.
- **Til:** o‘zbek tili, lotin yozuvi (`sh`, `ch`, `o‘`, `g‘`).
- **Turi:** qisqa, boshlanish–voqealar rivoji–yakun tuzilmasidagi hikoyalar.
- **Hajm:** 89 xil hikoya + pipeline testlari uchun qo‘shilgan 5 ta yozuv (jami 94 xom yozuv).
- **Manba fayl:** `data/source/hikoyalar_manba.jsonl`.
- **Kelib chiqishi:** loyiha uchun AI yordamida **yangi tuzilgan sintetik matnlar**. Hech qanday tashqi badiiy hikoya to‘g‘ridan-to‘g‘ri nusxalanmagan; TinyStories ishlatilmagan.
- **O‘quv maqsadi:** Midterm uchun ma’lumotlar tayyorlash bosqichlarini o‘zlashtirish, GPT/SLM uchun token ID binarlari yaratish.
- **Mazmun:** tabiat, maktab, sayohat, hunar, kuzatuv, ilmiy qiziqish, do‘stlik va mahalla voqealari.

## 2. Litsenziya va foydalanish huquqi (MUHIM)

**Belgilangan ochiq foydalanish sharti: CC0 1.0 Universal — huquqlar mavjud bo‘lgan darajada.**

Rasmiy manba: https://creativecommons.org/publicdomain/zero/1.0/

CC0 — huquq egasi mavjud mualliflik va unga aloqador huquqlardan qonun ruxsat etgan darajada voz kechishga mo‘ljallangan mexanizm. Ushbu loyiha tarkibidagi original sintetik matnlarga nisbatan mavjud bo‘lishi mumkin bo‘lgan huquqlar bo‘yicha foydalanishni cheklash maqsad qilinmaydi.

**Muhim ogohlantirish:** Matnlar generativ AI yordamida yaratilgan. Bunday matnlar uchun mualliflik huquqi mavjudligi, huquq kimga tegishli bo‘lishi yoki CC0 ni qo‘llash samarasi yurisdiksiyaga va inson hissasiga bog‘liq. Shuning uchun “barcha matnlar huquqiy jihatdan kafolatlangan public domain” degan da’vo qilinmaydi. Tashqi material qo‘shilsa, uning **alohida litsenziyasini** tekshirish zarur. CC0 boshqa shaxslarning shaxsiy hayot, maxfiylik yoki savdo belgisi kabi huquqlarini bekor qilmaydi.

Rasmiy shartlar: https://creativecommons.org/publicdomain/zero/1.0/legalcode.en

## 3. Dataset tarkibi va formati

Har qator bitta JSON obyekt:

```json
{
  "id": "H0001",
  "sarlavha": "Daraxt ostidagi daftar",
  "matn": "Mahalla kutubxonasi yonidagi chinor tagida Saida eski daftar topdi. ...",
  "til": "uz",
  "janr": "qisqa_hikoya",
  "manba": "loyiha_uchun_sintetik"
}
```

Barcha hikoyalar `UTF-8` kodlashda saqlanadi.

## 4. Pipeline va sun’iy test yozuvlari

Besh qo‘shimcha yozuvga ataylab noto‘g‘ri elementlar kiritilgan:

- `X0001` — takrorlangan matn;
- `X0002` — takrorlangan matnning biroz o‘zgargan varianti;
- `X0003` — reklama/URL;
- `X0004` — juda qisqa matn;
- `X0005` — **haqiqiy bo‘lmagan** email va telefon misoli (`example.com` domeni).

Bu orqali Filtering, Deduplication va PII Scrubbing haqiqatan ishlayotganini tekshirish mumkin. Test ma’lumotlarida haqiqiy insonning aloqa ma’lumoti kiritilmasligi kerak.

## 5. Cheklovlar

- Bu korpus **juda kichik**: kattaroq va sifatliroq LLM pre-training uchun yetarli emas.
- Hikoyalar sintetik bo‘lgani uchun uslubiy birxillik va takroriy iboralar bo‘lishi mumkin.
- Tarixiy va ilmiy faktlar rasmiy manba sifatida qabul qilinmasligi kerak.
- Regex PII detektori ism, uy manzili, pasport raqami va ko‘plab boshqa sezgir ma’lumotlarni aniqlamaydi.
- Datasetda uchinchi tomon web-scraping ma’lumotlari yo‘q.
- Litsenziya huquqiy masalalari bo‘yicha zarur bo‘lsa yurist bilan maslahat qiling.

## 6. Tashqi dataset bilan almashtirmoqchi bo‘lsangiz

1. Aniq URL, muallif va versiyani toping.
2. Litsenziyani va tijoriy/hosilaviy ishlatish huquqini tekshiring.
3. O‘sha saytdagi foydalanish shartlarini tekshiring; ruxsatsiz scraping qilmang.
4. Tasdiqlangan faylni `data/source` ga qo‘ying va `configs/data.yaml` ni yangilang.
5. PII tekshiruvi, sifat filtri va dedupni qayta ishga tushiring.
6. Dataset manbasi, sanasi, SHA256 va litsenziyasini shu READMEga yozing.

## 7. Kelajak uchun rivojlantirish

Ko‘proq hikoyalar, inson muharriri tekshiruvi, huquqiy jihatdan tekshirilgan korpus, tilni aniqlash, MinHash/LSH, PII NER (ism/manzil detektori), standart Hugging Face tokenizer va juda katta training dataset builder kiritilishi mumkin.
