# Förklaring av Chattbottar - Kapitel 10

## Vad är en chattbot?
En chattbot är som en robot som kan prata med dig! Precis som när du pratar med Siri eller Alexa, men denna kan du bygga själv.

---

## 1. Använda en chattbott genom API

**Vad händer här?**
- Vi ansluter till Googles AI (som heter Gemini) via internet
- Det är som att ringa till en smart robot och ställa en fråga
- Roboten svarar på din fråga!

**Exempel:** Du frågar "vem är zlatan?" och roboten berättar om Zlatan Ibrahimović.

---

## 2. Skapa en enkel applikation

**Vad händer här?**
- Nu bygger vi en egen chatt-app!
- Du kan skriva frågor i terminalen (som en chatt)
- Roboten svarar direkt på dina frågor
- Skriv "q" när du vill sluta prata

**Det är som:** Att ha en kompis som alltid svarar på dina frågor!

---

## 3. Retrieval Augmented Generation (RAG)

Detta är det smarta! Nu gör vi roboten ännu smartare genom att ge den en bok att läsa.

### 3.1 Läsa in PDF-fil

**Vad händer här?**
- Vi ger roboten en PDF-fil att läsa (som en bok)
- Roboten läser hela boken och kommer ihåg allt som står där
- Nu kan roboten svara på frågor om boken!

**Det är som:** Du ger roboten en lärobok, och den läser den och kan sedan svara på frågor om boken.

---

### 3.2 Chunking (Dela upp texten)

**Vad händer här?**
- Boken är för lång för roboten att läsa på en gång
- Så vi delar upp boken i små bitar (chunks)
- Varje bit är 1000 tecken lång
- Bitarna överlappar lite så inget viktigt går förlorat

**Det är som:** Att dela en lång pizza i små bitar så alla kan äta, men bitarna överlappar lite så ingen missar något gott!

---

### 3.3 Embeddings (Omvandla text till siffror)

**Vad händer här?**
- Datorer förstår inte ord, de förstår bara siffror
- Så vi omvandlar varje textbit till en lista med siffror
- Dessa siffror beskriver vad texten betyder

**Det är som:** Att översätta text till ett hemligt språk som bara datorn förstår, men där siffrorna fortfarande betyder samma sak!

---

### 3.4 Semantisk sökning (Hitta liknande texter)

**Vad händer här?**
- När du ställer en fråga, letar roboten efter textbitar som handlar om samma sak
- Den hittar texter som är lika i betydelse, inte bara samma ord
- Exempel: "Vad är RAG?" hittar text om RAG även om texten inte använder exakt samma ord

**Det är som:** Om du frågar "Vad är en bil?" så hittar roboten texter om "fordon", "auto", "bil" - allt som handlar om samma sak!

---

### 3.5 Generera bra svar med RAG

**Vad händer här?**
- Nu kombinerar vi allt!
- Du ställer en fråga
- Roboten hittar de mest relevanta textbitarna från boken
- Roboten svarar baserat bara på det som står i boken
- Om svaret inte finns i boken, säger roboten "Det vet jag inte"

**Det är som:** Du frågar något, roboten letar i boken, hittar rätt sida, och läser upp svaret för dig!

---

## 4. Evaluering (Kolla om roboten svarar rätt)

**Vad händer här?**
- Vi testar om roboten svarar korrekt
- Vi ger roboten frågor där vi vet svaret
- En annan robot kollar om svaret är rätt och ger poäng (0, 0.5 eller 1)
- Poäng 1 = helt rätt, 0.5 = delvis rätt, 0 = fel

**Det är som:** Ett prov för roboten! Vi kollar om den kan svara rätt på frågor vi redan vet svaret på.

---

## 5. Fördjupning

**Vad händer här?**
- Här finns länkar till mer avancerade saker du kan lära dig
- Om du vill bygga ännu mer avancerade chattbottar kan du läsa mer här

---

## Sammanfattning - Vad har vi lärt oss?

1. **Enkel chattbot:** Vi kan prata med en AI-robot via internet
2. **Smart chattbot:** Vi kan ge roboten en bok att läsa
3. **RAG-system:** Roboten kan svara på frågor baserat på boken den läst
4. **Testning:** Vi kan kolla om roboten svarar rätt

**Det är som att bygga en egen lärare-robot som kan läsa böcker och svara på dina frågor om böckerna!**

