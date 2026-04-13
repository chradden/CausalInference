# 01 – Einführung in Causal Inference (Kausale Inferenz)

## Worum geht es?

In diesem Modul wird das **Grundgerüst für kausales Denken** gelegt. Die zentrale Frage lautet: *Was bedeutet es eigentlich, dass etwas etwas anderes **verursacht**?*

---

## 1. Was ist Kausalität?

**Kausalität** bedeutet: Wenn ich gezielt etwas verändere (eine *Intervention* durchführe), dann passiert ein bestimmtes Ergebnis.

### Einfache Alltagsbeispiele

| Aktion | Ergebnis | Kausal? |
|--------|----------|---------|
| Du drückst die Snooze-Taste | Der Wecker hört auf zu klingeln | ✅ Ja – ohne Drücken klingelt er weiter |
| Du gießt Milch aus der Packung | Milch kommt in die Schüssel | ✅ Ja – ohne Gießen keine Milch |

### Schwierigere Beispiele

- Macht ein gesundes Frühstück dich im Alter gesünder? → **Unklar**, weil das Ergebnis erst viel später sichtbar wird und viele andere Einflüsse wirken.
- Was führt beim Football zum Touchdown – der Spielzug oder die Entscheidung weiterzuspielen?
- Was passiert, wenn ein Unternehmen die Preise erhöht? Was bewirkt ein neues Gesetz?

> **Kern-Erkenntnis:** Kausalität = „Wenn ich X verändere, passiert Y." Das unterscheidet sich fundamental von bloßer Beobachtung oder Korrelation.

---

## 2. Variablen messen

Um Kausalität **quantitativ** zu untersuchen, braucht man messbare Größen:

### Untersuchungseinheiten (Units of Analysis)
Die „Dinge", über die man Daten hat: Menschen, Länder, Firmen, Stadtblöcke, Dörfer.  
Die Gesamtheit aller Einheiten nennt man die **Population**.

### Die zwei zentralen Variablen

| Variable | Bedeutung | Beispiel |
|----------|-----------|---------|
| **Outcome-Variable** (Ergebnisvariable) | Das, was man verändern oder erklären will | Gesundheit, BIP, Kriminalität |
| **Treatment-Variable** (Maßnahme/Policy) | Das, womit man das Ergebnis beeinflussen will | Krankenversicherung, Polizeipräsenz, Eigentumsrechte |

### Messbarkeit

Abstrakte Konzepte müssen in **konkrete Zahlen** übersetzt werden:
- „Kriminalität" → Anzahl der Autodiebstähle pro Block
- „Wirtschaftliche Entwicklung" → BIP (Bruttoinlandsprodukt)
- „Hoffnung bei Kindern" → Merkmale aus Zeichnungen (z. B. ob ein Regenschirm gemalt wird)

---

## 3. Deskriptive Analyse und Variation

Bevor man kausal argumentiert, beschreibt man die Daten: Durchschnitt, Maximum, Minimum.

### Warum brauchen wir Variation?

Kausalität zu lernen heißt **vergleichen**: Menschen mit Behandlung vs. Menschen ohne Behandlung.

- Wenn alle Länder dieselben Eigentumsrechte hätten → keine Variation → kein Vergleich möglich → keine kausalen Erkenntnisse.
- Variation wird gemessen durch **Varianz** und **Standardabweichung** (größere Werte = mehr Unterschiede).

> **Beispiel „Top 5 Regrets of the Dying":** Aus einer Liste von Sterbenden-Wünschen allein kann man nicht schließen, ob man weniger arbeiten sollte – denn die **Vergleichsgruppe** (Menschen, die wenig gearbeitet haben) fehlt.

---

## 4. Der durchschnittliche Behandlungseffekt (ATE)

Da individuelle kausale Effekte für jede Person unterschiedlich sein können, konzentriert man sich oft auf den **Average Treatment Effect (ATE)**:

$$ATE = \text{Durchschnittliches Ergebnis mit Behandlung} - \text{Durchschnittliches Ergebnis ohne Behandlung}$$

| ATE-Wert | Interpretation (wenn höhere Werte besser sind) |
|----------|------------------------------------------------|
| Positiv | Die Maßnahme ist im Durchschnitt gut |
| Negativ | Die Maßnahme schadet im Durchschnitt |
| Nahe null | Die Maßnahme hat keinen nennenswerten Effekt |

> **Warnung:** Der Durchschnitt kann Extreme verbergen! Beispiel: Eine Tornado-Warn-App hat 4 von 5 Sternen im Schnitt – aber eine Person wurde nicht gewarnt. Durchschnitte allein reichen nicht immer.

---

## 5. Individuelle kausale Effekte und potenzielle Outcomes

### Unit-Level Causal Effect (individueller kausaler Effekt)

Für **dich** persönlich: Wie unterscheidet sich dein Ergebnis (z. B. Cholesterin) *mit* Behandlung (z. B. Krankenversicherung) von deinem Ergebnis *ohne* Behandlung – bei sonst gleichen Bedingungen?

### Potenzielle Outcomes

Für jede Person gibt es gedanklich **zwei mögliche Ergebnisse**:
1. **Y(1)** – Ergebnis mit Behandlung
2. **Y(0)** – Ergebnis ohne Behandlung

**Das fundamentale Problem der kausalen Inferenz:**
> Man kann **nie beide Ergebnisse gleichzeitig beobachten**. Eine Person kann nicht gleichzeitig versichert und nicht versichert sein. Eines der beiden Ergebnisse bleibt immer unsichtbar.

### Interaktionseffekte

Der kausale Effekt einer Maßnahme kann **davon abhängen, was sonst noch gilt**:
- Krankenversicherung bringt vielleicht nichts, wenn du dich ohnehin gesund ernährst.
- Aber sie hilft, wenn du dich schlecht ernährst.

→ Das nennt man **Heterogenität der Behandlungseffekte**: Verschiedene Menschen reagieren verschieden auf dieselbe Maßnahme.

---

## 6. Bedingte durchschnittliche Effekte (CATE)

Der **Conditional Average Treatment Effect (CATE)** ist der durchschnittliche Effekt **nur für eine bestimmte Gruppe**, z. B. nur für Männer oder nur für Frauen.

### Rechenbeispiel

| Person | Geschlecht | Y(1) mit Behandlung | Y(0) ohne Behandlung | Individueller Effekt |
|--------|-----------|---------------------|---------------------|---------------------|
| 1 | Mann | 40 | 30 | +10 |
| 2 | Mann | 20 | 20 | 0 |
| 3 | Frau | 10 | 15 | −5 |
| 4 | Frau | 30 | 30 | 0 |

- **ATE (gesamt):** (10 + 0 − 5 + 0) ÷ 4 = **+1,25** → positiv
- **CATE Männer:** (10 + 0) ÷ 2 = **+5** → deutlich positiv
- **CATE Frauen:** (−5 + 0) ÷ 2 = **−2,5** → negativ!

> **Erkenntnis:** Ein positiver Gesamteffekt bedeutet nicht, dass die Maßnahme für alle gut ist!

---

## 7. Confounder (Störvariablen)

Ein **Confounder** ist eine Variable, die:
1. sich zwischen der behandelten und der unbehandelten Gruppe **unterscheidet**, und
2. das **Ergebnis beeinflusst**.

Wenn Confounder vorhanden sind, kann eine beobachtete Korrelation **irreführend** sein.

> **Praxisregel:** Frage immer: *Gibt es Variablen, die zwischen den beiden Gruppen nicht konstant sind und die das Ergebnis beeinflussen könnten?*

---

## 8. Counterfactuals (Kontrafaktische Outcomes)

Nachdem wir Daten beobachtet haben, wird für jede Person eines der möglichen Ergebnisse zum **tatsächlichen** Ergebnis, das andere zum **kontrafaktischen** – dem Ergebnis, das eingetreten *wäre*, wenn die Behandlung anders gelaufen wäre.

> **Kausalität = Vergleich von Tatsächlichem und Gegenfaktischem.**

Da das Kontrafaktische nie beobachtbar ist, müssen Methoden der kausalen Inferenz es mit Annahmen **rekonstruieren**.

**Anschauliches Beispiel:** Winston Churchill schrieb einen Essay darüber, was passiert wäre, wenn Robert E. Lee die Schlacht von Gettysburg gewonnen hätte – eine kontrafaktische Geschichte, die strukturell der kausalen Forschung ähnelt.

---

## 9. Wie liest man ein empirisches Paper?

### Die 4 Kernfragen beim Überfliegen (ca. 5 Minuten)

1. **Was ist die Hauptfrage?** Welcher kausale Effekt wird untersucht?
2. **Was ist das Hauptergebnis?** Was wurde gefunden?
3. **Welche Daten werden verwendet?** Woher stammen sie?
4. **Welche Methode wird eingesetzt?** Welche Identifikationsstrategie?

### Empfohlene Lesereihenfolge

1. **Empirische Strategie & Hauptergebnisse** → Die wichtigste Tabelle/Grafik finden
2. **Datenquellen & Variablendefinitionen** → Was genau wurde gemessen?
3. **Robustheitschecks** → Halten die Ergebnisse auch unter anderen Annahmen?
4. **Einleitung, Theorie, Schluss** → Breiterer Kontext (kann übersprungen werden, wenn man das Thema kennt)

---

## Kernbotschaften

1. **Kausalität ≠ Korrelation** – Kausalität fragt immer: Was passiert, wenn ich etwas gezielt verändere?
2. **Das fundamentale Problem:** Für jede Person bleibt ein relevantes Ergebnis immer unbeobachtet.
3. **Die gesamte kausale Forschung** ist der Versuch, dieses fehlende Gegenfaktum mithilfe von Daten, Variation, Vergleichsgruppen und Annahmen zu rekonstruieren.

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **Kausalität** | Eine Veränderung A *verursacht* eine Veränderung B |
| **Outcome-Variable** | Die Ergebnisgröße (z. B. Gesundheit, BIP) |
| **Treatment-Variable** | Die Maßnahme/Intervention (z. B. Versicherung) |
| **Population** | Gesamtheit aller untersuchten Einheiten |
| **ATE** | Average Treatment Effect – durchschnittlicher Behandlungseffekt |
| **CATE** | Conditional ATE – bedingter Durchschnittseffekt für eine Untergruppe |
| **Confounder** | Störvariable, die den kausalen Vergleich verzerrt |
| **Counterfactual** | Das Ergebnis, das eingetreten wäre, wenn etwas anders gelaufen wäre |
| **Potenzielle Outcomes** | Die zwei hypothetischen Ergebnisse Y(1) und Y(0) |
| **Variation** | Unterschiede in der Treatment-Variable – nötig für kausale Analyse |
