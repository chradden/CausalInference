# 03 – Machine Learning zur Schätzung durchschnittlicher Behandlungseffekte (ATEs)

## Worum geht es?

Dieses Modul zeigt, wie man den **durchschnittlichen kausalen Effekt (ATE)** einer Maßnahme mit Hilfe von Machine Learning schätzen kann. Es werden drei Ansätze vorgestellt – von einfach bis robust – und ihre Stärken und Schwächen verglichen.

---

## 1. Das Ausgangsproblem: Fehlende Daten

### Potenzielle Outcomes

Für jede Person gibt es zwei hypothetische Ergebnisse:
- **Y(1)** – das Ergebnis *mit* Behandlung
- **Y(0)** – das Ergebnis *ohne* Behandlung

Der **kausale Effekt** für eine Person ist die Differenz: τᵢ = Y(1) − Y(0).

**Das Problem:** Wir sehen immer nur *eines* der beiden Ergebnisse – je nachdem, ob die Person tatsächlich behandelt wurde oder nicht.

### Randomisierte Experimente als Goldstandard

Wenn die Behandlung **zufällig** zugewiesen wird:
- Behandelte und Kontrollgruppe sind im Durchschnitt identisch
- Einfacher Mittelwertvergleich liefert den ATE: $\hat{\tau} = \bar{Y}_{\text{behandelt}} - \bar{Y}_{\text{kontrolle}}$

### Selection Bias (Selektionsverzerrung)

Ohne Zufallszuweisung: Personen, die behandelt werden, unterscheiden sich **systematisch** von denen, die nicht behandelt werden.

> **Beispiel:** Ärzte geben kränkeren Patienten eher ein Medikament. Wenn man dann die Behandelten mit den Nicht-Behandelten vergleicht, sieht es so aus, als sei das Medikament schlecht – weil die Behandelten von vornherein kränker waren.

---

## 2. Strategie 1: Regression Adjustment (Regressionsanpassung)

### Grundidee

Wenn die Behandlung „so gut wie zufällig" ist, **nachdem man bestimmte Merkmale X berücksichtigt hat** (= Unconfoundedness-Annahme), kann man:

1. Für die **Behandlungsgruppe** schätzen: μ₁(X) = E[Y | X, behandelt]
2. Für die **Kontrollgruppe** schätzen: μ₀(X) = E[Y | X, nicht behandelt]
3. ATE berechnen als: $\hat{\tau} = \frac{1}{n}\sum [\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)]$

### Mit linearer Regression
- **Vorteil:** Schnell, leicht verständlich, gute Konvergenzrate ($\sqrt{n}$)
- **Nachteil:** Wenn das lineare Modell falsch ist → **komplett falsche Ergebnisse** (inkonsistent)

### Mit Machine Learning (z. B. Random Forest)
- **Vorteil:** Flexibel, passt sich an nichtlineare Zusammenhänge an
- **Nachteil:** Konvergiert **langsamer** ($n^{-1/4}$ statt $n^{-1/2}$) → Unsicherheit ist in endlichen Stichproben zu groß für zuverlässige Konfidenzintervalle

> **Trade-off:** Lineare Regression ist „nie ganz falsch, aber nie ganz richtig" – ML ist flexibler, hat aber langsamere Konvergenz.

---

## 3. Strategie 2: Inverse Propensity Weighting (IPW)

### Was ist der Propensity Score?

Der **Propensity Score** $e(X)$ ist die Wahrscheinlichkeit, behandelt zu werden, gegeben die Merkmale X:

$$e(X) = P(W = 1 \mid X)$$

### Die IPW-Idee

Man **gewichtet** jede Beobachtung **umgekehrt** mit ihrer Wahrscheinlichkeit, die Behandlung erhalten zu haben, die sie tatsächlich bekommen hat:

$$\hat{\tau}_{IPW} = \frac{1}{n}\sum \left[\frac{W_i \cdot Y_i}{e(X_i)} - \frac{(1-W_i) \cdot Y_i}{1 - e(X_i)}\right]$$

### Intuition

- Eine behandelte Person mit **niedrigem** Propensity Score (also: es war unwahrscheinlich, dass sie behandelt wird) bekommt ein **hohes** Gewicht.
- So wird eine Art „Pseudo-Randomisierung" erzeugt.

### Problem mit geschätzten Propensity Scores

Wenn man $e(X)$ mit ML schätzt und einsetzt:
- Die ML-Schätzfehler **heben sich nicht auf**
- Der Gesamtfehler ist von der Ordnung $n^{-1/4}$ statt $n^{-1/2}$
- Konfidenzintervalle werden **zu eng** und damit **unzuverlässig**

> **Ergebnis:** Naives IPW mit ML-geschätzten Propensity Scores funktioniert nicht gut genug.

---

## 4. Strategie 3: Doubly Robust Estimation (AIPW) – Die Lösung

### Was ist AIPW?

Der **Augmented Inverse Propensity Weighted (AIPW)** Schätzer kombiniert Regression Adjustment und IPW auf eine clevere Weise:

$$\hat{\tau}_{AIPW} = \frac{1}{n}\sum \left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{W_i(Y_i - \hat{\mu}_1(X_i))}{e(X_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(X_i))}{1-e(X_i)}\right]$$

### Zwei Komponenten

| Komponente | Was sie tut |
|------------|-------------|
| **Regressionsteil** | Direkte Vorhersage: μ₁(X) − μ₀(X) |
| **IPW-Korrekturteil** | Korrigiert den Fehler der Regressionsschätzung mit Hilfe der Residuen |

### Warum das funktioniert: Double Robustness

Das Geniale an AIPW:
- Fehler in der Regressionsschätzung und Fehler in der Propensity-Score-Schätzung **heben sich in erster Ordnung auf**
- Beide dürfen langsam konvergieren (z. B. $n^{-1/4}$)
- Der Gesamtschätzer konvergiert trotzdem schnell: $\sqrt{n}$-Rate
- **Normale Verteilung** → valide Konfidenzintervalle und Hypothesentests!

### Voraussetzungen

1. **Unconfoundedness:** Behandlung ist „so gut wie zufällig" gegeben X
2. **Overlap:** Jede Person hat eine Wahrscheinlichkeit von behandelt zu werden, die nicht zu nahe an 0 oder 1 liegt: $0 < e(X) < 1$
3. **Cross-Fitting:** Die Hilfsschätzungen (μ und e) werden auf **anderen Daten** berechnet als die Hauptschätzung

---

## 5. Cross-Fitting erklärt

### Warum braucht man das?

Wenn man dieselben Daten benutzt, um μ̂(X) und ê(X) zu schätzen UND um den ATE zu berechnen, kann sich das Modell „selbst bestätigen" → verzerrte Ergebnisse.

### Wie es funktioniert

1. Teile die Daten in $K$ Teile (z. B. 5).
2. Für jeden Teil: Schätze μ̂ und ê auf den **anderen** Teilen.
3. Berechne den AIPW-Schätzer auf dem aktuellen Teil mit den „fremden" Schätzungen.
4. Mittele über alle Teile.

> **Ergebnis:** Unverzerrt und effizient – die ML-Schätzfehler „vergiften" den Hauptschätzer nicht.

---

## 6. Zusammenfassung der drei Ansätze

| Methode | Stärke | Schwäche | Konvergenz |
|---------|--------|----------|------------|
| **Regression Adjustment** | Einfach, intuitiv | Bricht zusammen bei Modellfehlspezifikation | $\sqrt{n}$ nur wenn Modell korrekt |
| **IPW** | Nutzt Propensity Score | Extrem empfindlich gegenüber Propensity-Fehlern | $n^{-1/4}$ mit ML |
| **AIPW (Doubly Robust)** | Kombiniert beide; Fehler heben sich auf | Etwas komplexer umzusetzen | $\sqrt{n}$ auch mit ML ✅ |

> **Empfehlung:** Für die Praxis ist AIPW mit Cross-Fitting der beste Ansatz, wenn ML für die Hilfsschätzungen verwendet wird.

---

## Kernbotschaften

1. **Zufällige Zuweisung** ist der Goldstandard, aber oft nicht verfügbar.
2. **Regression Adjustment mit ML** konvergiert zu langsam für zuverlässige Inferenz.
3. **IPW mit ML** ist instabil, weil Propensity-Score-Fehler sich aufschaukeln.
4. **AIPW (Doubly Robust)** löst das Problem: Fehler heben sich auf → schnelle Konvergenz + valide Inferenz.
5. **Cross-Fitting** ist entscheidend dafür, dass alles funktioniert.

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **ATE** | Average Treatment Effect – durchschnittlicher Behandlungseffekt |
| **Potenzielle Outcomes** | Y(1) und Y(0) – hypothetische Ergebnisse mit/ohne Behandlung |
| **Selection Bias** | Verzerrung, weil sich behandelte und unbehandelte Gruppen systematisch unterscheiden |
| **Unconfoundedness** | Annahme: Behandlung ist „so gut wie zufällig", wenn man X kontrolliert |
| **Propensity Score** | e(X) = Wahrscheinlichkeit, behandelt zu werden, gegeben X |
| **IPW** | Inverse Propensity Weighting – Gewichtung durch inverse Behandlungswahrscheinlichkeit |
| **AIPW** | Augmented IPW – kombiniert Regression und IPW; doubly robust |
| **Cross-Fitting** | Hilfsschätzungen auf anderen Daten berechnen als den Hauptschätzer |
| **Overlap** | Jede Person hat realistische Chance auf Behandlung und Nicht-Behandlung |
| **Double Robustness** | Fehler in den Hilfsschätzungen heben sich in erster Ordnung auf |
