# 06 – Verlustfunktionen für kausale Inferenz & Policy Learning

## Worum geht es?

Die vorherigen Module haben gezeigt, wie man Behandlungseffekte schätzt – erst Durchschnittseffekte (Modul 03), dann heterogene Effekte (Module 04–05). Dieses Modul macht den **letzten großen Schritt**: Wie verbindet man kausale Inferenz mit dem Kernkonzept von Machine Learning – der **Verlustfunktion (Loss Function)**? Außerdem: Wie **bewertet** man CATE-Schätzer und wie lernt man optimale **Behandlungspolitiken**?

---

## 1. Machine Learning = Aufgabe + Verlustfunktion + Daten

### Die universelle ML-Formel

Jedes ML-Problem hat drei Zutaten:

| Zutat | Beschreibung | Beispiel: Vorhersage | Beispiel: Kausale Inferenz |
|-------|-------------|---------------------|---------------------------|
| **Aufgabe** | Was soll erreicht werden? | Y vorhersagen | Behandlungseffekt τ(X) schätzen |
| **Verlustfunktion** | Was bestraft Fehler? | (Y − μ̂(X))² | R-Loss (siehe unten) |
| **Daten** | Woraus wird gelernt? | Training/Test-Split | X, Y, W + Cross-Fitting |

### Warum Verlustfunktionen so mächtig sind

Sobald man die **richtige Verlustfunktion** hat, kann man jedes beliebige ML-Verfahren einsetzen:
- **LASSO**: Minimiere Loss + L1-Penalty
- **Boosting**: Minimiere Loss schrittweise
- **Neuronale Netze**: Minimiere Loss per Gradientenabstieg
- **Random Forests / Bäume**: Minimiere Loss per Split-Auswahl

> **Kernidee:** Tausche einfach die Verlustfunktion aus – der Rest des ML-Frameworks bleibt gleich!

---

## 2. Von Vorhersage zu Kausalität: Die R-Loss-Funktion

### Bei Vorhersage (bekannt)

$$L_{\text{pred}} = (Y_i - \hat{\mu}(X_i))^2$$

Minimiere den quadrierten Vorhersagefehler → ergibt die beste Vorhersagefunktion μ̂(X).

### Für den Behandlungseffekt: Robinson's Loss (R-Loss)

Aus Robinson's Transformation (Modul 05) wissen wir:

$$Y - m(X) = \tau(X) \cdot (W - e(X)) + \varepsilon$$

Daraus folgt eine natürliche Verlustfunktion:

$$L_R = \bigl(Y_i - \hat{m}(X_i) - \hat{\tau}(X_i) \cdot (W_i - \hat{e}(X_i))\bigr)^2$$

### Was steckt drin?

| Symbol | Bedeutung |
|--------|-----------|
| $Y_i$ | Beobachtetes Outcome |
| $m(X)$ | Erwartetes Y, wenn man Behandlung ignoriert |
| $e(X)$ | Propensity Score (Behandlungswahrscheinlichkeit) |
| $\tau(X)$ | Der Behandlungseffekt, den wir schätzen wollen |
| $W_i$ | Tatsächliche Behandlung (0 oder 1) |

### Intuition

> **Stell dir vor:** Du nimmst das bereinigte Outcome $Y^* = Y - m(X)$ und die bereinigte Behandlung $W^* = W - e(X)$. Dann ist τ(X) genau der Koeffizient, der Y\* am besten durch W\* vorhersagt – lokal für jeden Wert von X.

---

## 3. Der R-Learner – Universeller CATE-Schätzer

### Der Algorithmus

1. **Schritt 1:** Schätze m̂(X) = E[Y | X] (z. B. mit Boosting)
2. **Schritt 2:** Schätze ê(X) = P(W=1 | X) (z. B. mit Boosting)
3. **Schritt 3:** Minimiere die R-Loss-Funktion über τ(X) mit beliebigem ML-Verfahren:

$$\hat{\tau} = \arg\min_\tau \frac{1}{n}\sum_{i=1}^n \bigl(Y_i - \hat{m}(X_i) - \tau(X_i) \cdot (W_i - \hat{e}(X_i))\bigr)^2 + \lambda \cdot \text{Reg}(\tau)$$

### Warum „R-Learner"?

Das „R" steht für **Robinson** (nach Robinson's Transformation). Der R-Learner:
- ist ein **allgemeines Framework**, kein einzelner Algorithmus
- Funktioniert mit LASSO, Boosting, Deep Nets, Random Forests, …
- Hat die gleichen **Double-Robustness-Eigenschaften** wie AIPW und Robinson's Methode

### Vorteile gegenüber T/S/X-Learnern

| Eigenschaft | T/S/X-Learner | R-Learner |
|-------------|---------------|-----------|
| Braucht explizit den Propensity Score | ❌ (ignoriert ihn oft) | ✅ (eingebaut) |
| Confounding-robust | ❌ oder teilweise | ✅ |
| Kann schneller für τ konvergieren als für Hilfsschätzungen | ❌ | ✅ |
| Beliebiges ML als Basis nutzbar | ✅ | ✅ |

---

## 4. Praxisbeispiel: Get-out-the-Vote-Anrufe

### Setup

- **Reale Daten** mit Confounding (unterschiedliche Wahlkreise, verschiedene Anrufwahrscheinlichkeiten)
- Der wahre Behandlungseffekt eines einzelnen Anrufs ist quasi **null**
- Es werden kleine **synthetische Effekte** eingespielt („Semi-Synthetic")

### Ablauf

1. **m̂ und ê** schätzen: Cross-Validation wählt Boosting als bestes Verfahren
2. **τ̂** mit R-Loss schätzen: Cross-Validation wählt LASSO
3. **Ergebnis:** Der R-LASSO rekonstruiert den eingespielten Effekt zuverlässig

> **Wichtig:** Man kann in den einzelnen Schritten **verschiedene ML-Methoden** kombinieren – Boosting für die Hilfsschätzungen, LASSO für den Effekt.

---

## 5. CATE-Schätzer bewerten und vergleichen

### Das Problem

In der Praxis kennt man den wahren Behandlungseffekt **nicht**. Wie beurteilt man dann, ob ein CATE-Schätzer gut ist?

### Methode 1: R-Loss-Differenz (ΔR)

Vergleiche den R-Loss eines Schätzers mit dem R-Loss einer **konstanten** Schätzung (kein Heterogenität):

$$\Delta R = R\text{-Loss}(\hat{\tau}) - R\text{-Loss}(\bar{\tau})$$

| ΔR-Wert | Interpretation |
|---------|----------------|
| **Negativ** | τ̂ findet echte Heterogenität → besser als konstant |
| **Nahe null** | τ̂ bringt kaum Mehrwert gegenüber Durchschnittseffekt |
| **Positiv** | τ̂ ist **schlechter** als einfach einen konstanten Effekt zu schätzen! |

> **Achtung:** Der absolute R-Loss ist riesig und für alle Methoden ähnlich. Nur die **Differenz** ist informativ!

### Methode 2: Gruppierung nach Quartilen

1. Teile Personen anhand von τ̂(X) in Quartile ein (niedrig → hoch)
2. Schätze den **tatsächlichen** ATE pro Quartil auf gehaltenen Daten
3. Prüfe: Steigt der Effekt wirklich von Quartil 1 zu 4?

### Methode 3: Kalibrierungs-Regression

Regressiere den tatsächlichen Behandlungseffekt (auf gehaltenen Daten) auf τ̂(X):

| Koeffizient β | Interpretation |
|---------------|----------------|
| **β ≈ 1** | Gut kalibriert – geschätzte Effekte passen zur Realität |
| **β > 1** | Underfitting – echte Heterogenität ist stärker als geschätzt |
| **β ≈ 0** | Keine echte Heterogenität gefunden |
| **β < 0** | Katastrophe – Schätzer dreht die Richtung um! |

### Methode 4: QINI-Kurve

Nützlich, wenn man Behandlung **priorisieren** will (z. B. wegen Budgetbeschränkung):

- **X-Achse:** Anteil der behandelten Personen (0 % bis 100 %)
- **Y-Achse:** Kumulativer Gewinn durch Behandlung
- Vergleiche Targeting nach τ̂(X) mit zufälliger Zuweisung
- Liegt die Kurve **über** der Zufallslinie → Targeting lohnt sich

---

## 6. Policy Learning – Optimale Behandlungspolitiken

### Was ist eine Policy?

Eine **Policy** π(X) ist eine Entscheidungsregel: Für jede Person mit Merkmalen X → Behandlung (1) oder keine Behandlung (0).

### Unterschied zu CATE-Schätzung

| | CATE-Schätzung | Policy Learning |
|--|----------------|-----------------|
| **Ziel** | τ(X) möglichst genau schätzen | Möglichst hohen Nutzen erzielen |
| **Output** | Eine Zahl (τ̂) pro Person | Eine Entscheidung (0/1) pro Person |
| **Loss** | Quadratischer Fehler auf τ-Skala | Negative Wohlfahrt |
| **Typ** | Regressions-Problem | Klassifikations-Problem |

### Die Wohlfahrt einer Policy

$$V(\pi) = E[Y_i(\pi(X_i))]$$

> Die erwartete durchschnittliche Outcome-Qualität, wenn man nach Policy π behandelt.

### Policy-Loss mit Double-Robust-Scores

Statt IPW mit bekannten Propensity Scores verwendet man die **Double-Robust-Scores** Γ̂ᵢ aus dem AIPW-Schätzer:

$$\hat{V}(\pi) = \frac{1}{n}\sum_{i=1}^n \hat{\Gamma}_i \cdot (2\pi(X_i) - 1)$$

Wobei $\hat{\Gamma}_i$ jeweils ein verrauschter Proxy für den Behandlungseffekt der i-ten Person ist.

### Optimierung über Entscheidungsbäume

- In der Praxis wird π über die Klasse der **Entscheidungsbäume** der Tiefe k optimiert
- Das R-Paket `policytree` implementiert dies
- Ergebnis: Ein einfacher, interpretierbarer Baum als Behandlungsregel

---

## 7. Praxisbeispiel: GAIN-Wohlfahrtsprogramm

### Setup

- **GAIN** = Arbeitsvermittlungsprogramm für Sozialhilfeempfänger in Kalifornien
- Daten aus **4 verschiedenen Counties** (Alameda, LA, Riverside, San Diego)
- Unterschiedliche Randomisierungsanteile und Zielgruppen → zusammengeführt eine **Beobachtungsstudie**

### CATE-Vergleich mit R-Loss

| Methode | ΔR | Bewertung |
|---------|-----|-----------|
| T-Forest | > 0 | Schlechter als konstant |
| S-Forest | > 0 | Schlechter als konstant |
| X-Forest | < 0, signifikant | Findet echte Heterogenität ✅ |
| Causal Forest | < 0, signifikant | Findet echte Heterogenität ✅ |

### Policy-Baum (Tiefe 2)

```
War Person vor 3 Quartalen beschäftigt?
├── Ja → Behandlung zuweisen ✅
└── Nein → Highschool-Abschluss?
    ├── Ja → Behandlung zuweisen ✅
    └── Nein → Keine Behandlung
```

### Ethische Aspekte

- Variablen wie **Geschlecht** und **Ethnie** werden bewusst **ausgeschlossen**
- Der Entscheidungsbaum ist **einfach und transparent** → Stakeholder können ihn prüfen
- Entscheidung, ob die Policy fair ist, liegt bei den **Entscheidungsträgern**, nicht bei Statistikern

---

## 8. Zusammenfassung: Der vollständige ML-Workflow für kausale Inferenz

```
┌─────────────────────────────────────────────┐
│  1. Hilfsschätzungen (Nuisance Estimation)  │
│     m̂(X) = E[Y|X]  und  ê(X) = P(W=1|X)   │
│     → Beliebiges ML + Cross-Fitting        │
├─────────────────────────────────────────────┤
│  2. Kausale Schätzung                       │
│     Option A: ATE → AIPW (Modul 03)        │
│     Option B: CATE → R-Learner / Causal    │
│               Forest (Module 05–06)         │
│     Option C: Policy → Policy Tree          │
├─────────────────────────────────────────────┤
│  3. Evaluation                               │
│     R-Loss-Differenz, Quartile, Kalibrie-   │
│     rung, QINI-Kurve                        │
└─────────────────────────────────────────────┘
```

---

## Kernbotschaften

1. **Verlustfunktionen** sind das Bindeglied zwischen ML und kausaler Inferenz – tausche die Loss-Funktion aus und nutze beliebige ML-Methoden.
2. Die **R-Loss-Funktion** (aus Robinson's Transformation) ist die richtige Loss-Funktion für CATE-Schätzung; sie hat Double-Robustness-Eigenschaften.
3. **R-Learner** = Universelles Framework: Schätze m̂, ê, dann minimiere R-Loss mit beliebigem ML.
4. **Evaluation** ist entscheidend: Nicht der absolute R-Loss zählt, sondern die **Differenz** zum Baseline.
5. **Policy Learning** ist ein anderes Problem als CATE-Schätzung – es optimiert Entscheidungen (0/1), nicht Schätzungen.
6. **Transparenz und Ethik** sind bei Policy Learning besonders wichtig – Entscheidungsbäume helfen, weil sie verständlich sind.

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **Verlustfunktion (Loss)** | Maß dafür, wie schlecht eine Schätzung/Vorhersage ist; wird minimiert |
| **R-Loss** | Quadratischer Fehler nach Robinson's Transformation – angepasst für CATE |
| **R-Learner** | Framework: Minimiere R-Loss mit beliebigem ML-Verfahren |
| **Policy** | Entscheidungsregel π(X) → behandeln (1) oder nicht (0) |
| **Policy Learning** | Lernen der besten Entscheidungsregel aus Daten |
| **Wohlfahrt V(π)** | Erwarteter Nutzen, wenn man Policy π anwendet |
| **Double-Robust-Scores** | Γ̂ᵢ – verrauschte Proxies für individuelle Behandlungseffekte |
| **QINI-Kurve** | Kosten-Nutzen-Diagramm für priorisierte Behandlungszuweisung |
| **ΔR (R-Loss-Differenz)** | Vergleich eines Schätzers mit dem konstanten Baseline-Schätzer |
| **Kalibrierung** | Prüfung, ob geschätzte Effekte mit realen Effekten übereinstimmen |
| **Policy Tree** | Entscheidungsbaum als optimale Behandlungsregel |
| **Semi-Synthetic** | Reale Daten + künstlich eingefügte Behandlungseffekte zur Evaluation |
