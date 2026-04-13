# 04 – Machine Learning für heterogene Behandlungseffekte: Causal Trees & Forests

## Worum geht es?

Im vorherigen Modul ging es um den **durchschnittlichen** Behandlungseffekt (ATE). Aber verschiedene Menschen reagieren verschieden auf dieselbe Maßnahme! Dieses Modul zeigt, wie man mit ML herausfinden kann, **für wen** eine Behandlung besonders gut oder schlecht wirkt – der sogenannte **bedingte durchschnittliche Behandlungseffekt (CATE)**.

---

## 1. Warum Heterogenität wichtig ist

### Nicht jeder profitiert gleich

- Ein Medikament kann bei manchen Patienten hervorragend wirken und bei anderen nichts bringen.
- Ein Bildungsprogramm kann für bestimmte Gruppen stark helfen, für andere kaum.
- Eine Marketingaktion trifft manche Kunden ins Herz, andere ignorieren sie.

### Was wir lernen wollen

Statt nur **einen** Durchschnittswert zu kennen, wollen wir:
- **Τ(X) = E[Y(1) − Y(0) | X]** – den bedingten durchschnittlichen Behandlungseffekt als Funktion von Personen-Merkmalen X
- Herausfinden, **welche Merkmale** die Heterogenität treiben
- **Personalisierte Empfehlungen** ableiten (wer sollte behandelt werden, wer nicht?)

### Herausforderungen
- Viel **schwieriger** als den ATE zu schätzen – braucht deutlich mehr Daten
- Individuelle Behandlungseffekte τᵢ sind **nie direkt beobachtbar**
- Schwache Signale in vielen Anwendungen → Gefahr von Scheinmustern

---

## 2. Causal Trees – Bäume für kausale Effekte

### Grundidee

Ein **Regressionbaum** teilt den Merkmalsraum in Bereiche und sagt in jedem Bereich einen Wert vorher. Ein **Causal Tree** macht das Gleiche, aber statt Outcomes vorherzusagen, schätzt er **Behandlungseffekte** in jedem Bereich.

### Wie es funktioniert

```
Einkommen > 50.000?
├── Ja → Bildung > 12 Jahre?
│   ├── Ja → Behandlungseffekt: +8
│   └── Nein → Behandlungseffekt: +2
└── Nein → Politische Einstellung konservativ?
    ├── Ja → Behandlungseffekt: −3
    └── Nein → Behandlungseffekt: +1
```

### Das Split-Kriterium

Statt den Vorhersagefehler für Y zu minimieren, maximiert ein Causal Tree den **Unterschied in den Behandlungseffekten** zwischen den Kinderknoten:

$$\text{Split-Qualität} = |\hat{\tau}_{\text{links}} - \hat{\tau}_{\text{rechts}}|^2 \times n_{\text{links}} \times n_{\text{rechts}}$$

> **Ziel:** Gruppen finden, die sich möglichst stark in ihrem Behandlungseffekt unterscheiden.

---

## 3. Honesty: Die Trennung von Baum-Konstruktion und Schätzung

### Das Problem

Wenn man **dieselben Daten** benutzt, um den Baum zu bauen UND die Behandlungseffekte in den Blättern zu schätzen:
- Der Baum sucht gezielt nach großen Unterschieden → **Overfitting**
- Geschätzte Effekte sind **nach oben verzerrt**
- Konfidenzintervalle sind **zu eng** und unzuverlässig

### Die Lösung: Sample Splitting (Honest Trees)

| Schritt | Datenteil | Aufgabe |
|---------|-----------|---------|
| 1 | **Training Sample** | Baum wachsen lassen (Splits finden) |
| 2 | **Estimation Sample** | Behandlungseffekte in den Blättern schätzen |

> **Ergebnis:** Die Schätzungen sind unverzerrt und es gibt valide Konfidenzintervalle für jeden Behandlungseffekt.

---

## 4. Anwendungsbeispiele

### Beispiel 1: „Welfare" vs. „Hilfe für Arme" (General Social Survey)

**Frage:** Befürworten Menschen Sozialausgaben mehr, wenn man sie „Hilfe für Arme" statt „Welfare" nennt?

**Ergebnisse des Causal Tree:**
- Der Baum splittet auf Einkommen, Bildung und politische Einstellung
- **Konservativere Menschen** reagieren stärker auf die Wortwahl (der Framing-Effekt ist größer)
- Honest Estimation liefert verlässliche Konfidenzintervalle

### Beispiel 2: Microsoft Bing (Suchposition)

**Frage:** Wie stark ändert sich die Klickrate, wenn man Suchergebnisse in ihrer Position verschiebt?

**Ergebnisse:**
- Die oberen 20% der Suchanfragen zeigen **starke Positionseffekte**
- Bild-/Promi-Suchen zeigen **kleine Effekte** (weniger Klicks insgesamt)
- Der Baum findet 20 Blätter mit unterschiedlichen Effektstärken
- Honest Estimation zeigt weniger Variabilität als adaptive Schätzung

---

## 5. Causal Forests – Von einem Baum zum Wald

### Problem mit einzelnen Bäumen

- Ein einzelner Baum hat **scharfe Sprünge** an den Grenzen der Blätter
- Kleine Änderungen im Training können einen **ganz anderen Baum** ergeben
- Die Schätzung ist **nicht glatt**

### Lösung: Viele Bäume mitteln → Causal Forest

1. Ziehe viele leicht verschiedene **Unterstichproben** der Daten
2. Baue auf jeder einen ehrlichen Causal Tree (mit Sample Splitting)
3. **Mittele** die Vorhersagen aller Bäume

### Ergebnis
- **Glatte** Schätzung des Behandlungseffekts τ̂(X)
- Jeder Baum wird auf anderen Daten trainiert → weniger Overfitting
- Der Wald liefert **adaptive Gewichte** α(X) für jede Beobachtung
- Asymptotisch normalverteilt → **valide Konfidenzintervalle**

### Adaptives Nearest-Neighbor-Verfahren

Man kann den Causal Forest als intelligentes Nachbarschaftsverfahren verstehen:
- Statt die k nächsten Nachbarn gleich zu gewichten (was in hohen Dimensionen versagt)
- Gewichtet der Forest Nachbarn **adaptiv** – basierend auf den Variablen, die für den Behandlungseffekt wirklich relevant sind

---

## 6. Anwendungsbeispiel: FAFSA „Nudge" Experiment

**Hintergrund:** Textnachrichten erinnern Studenten daran, ihr Finanzhilfe-Formular (FAFSA) einzureichen.

**Ergebnisse:**
- Signal ist insgesamt **schwach** (verrauschte Daten)
- **Einschreibungsstatus** ist der wichtigste Prädiktor für Heterogenität
- Bereits eingeschriebene Studenten reagieren **stärker** auf den Nudge
- Zeigt Unterschied zwischen „explorativer Heterogenitätssuche" und vorab definierten Hypothesen

### Targeting-Analyse

- Menschen mit hohem geschätztem τ̂(X) gezielt behandeln bringt **manchmal** nur marginale Gewinne
- Manchmal ist es besser, Menschen zu behandeln, die „ohnehin fast reagiert hätten"
- **Kontext ist entscheidend** für die Interpretation

---

## 7. Zusammenfassung: Bäume vs. Wald

| Eigenschaft | Causal Tree | Causal Forest |
|-------------|-------------|---------------|
| **Flexibilität** | Stückweise konstant | Glatt |
| **Interpretierbarkeit** | Hoch (lesbarer Baum) | Mittel (viele Bäume) |
| **Stabilität** | Gering (Baum ändert sich leicht) | Hoch (gemittelt) |
| **Konfidenzintervalle** | Ja (mit Honesty) | Ja (asymptotisch normal) |
| **Overfitting-Risiko** | Höher | Niedriger |

---

## Kernbotschaften

1. **Heterogenität ist wichtig** – der Durchschnittseffekt kann eine Gruppe verbergen, der die Behandlung schadet.
2. **Causal Trees** partitionieren den Merkmalsraum, um Gruppen mit unterschiedlichen Behandlungseffekten zu finden.
3. **Honesty (Sample Splitting)** ist entscheidend: Baumstruktur und Effektschätzung auf verschiedenen Daten.
4. **Causal Forests** glätten die Schätzung durch Mittelung vieler ehrlicher Bäume → stabiler und statistisch valide.
5. Die Methoden sind **explorativ** – Ergebnisse müssen im Kontext interpretiert werden und ersetzen keine vorab geplanten Hypothesen.

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **CATE** | Conditional Average Treatment Effect – bedingter Durchschnittseffekt für eine Untergruppe |
| **Heterogenität** | Behandlungseffekte unterscheiden sich zwischen Personen |
| **Causal Tree** | Entscheidungsbaum, der Behandlungseffekte statt Outcomes schätzt |
| **Honest Estimation** | Baumstruktur und Effektschätzung auf verschiedenen Daten |
| **Causal Forest** | Mittelung vieler ehrlicher Causal Trees |
| **Adaptive Gewichte** | Der Forest gewichtet Nachbarn basierend auf relevanten Merkmalen |
| **Targeting** | Gezielte Zuweisung der Behandlung an Gruppen mit hohem Effekt |
| **Nudge** | Sanfter Anstoß (z. B. Erinnerungs-SMS) zur Verhaltensänderung |
| **Subsampling** | Ziehung von Unterstichproben für einzelne Bäume im Forest |
