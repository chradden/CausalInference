# 05 – Robuste Schätzung von Behandlungseffekten in Beobachtungsstudien

## Worum geht es?

Bisher wurde oft vorausgesetzt, dass die Behandlung **zufällig** zugewiesen wird (Experiment). In der Praxis ist das selten der Fall – Menschen „wählen" ihre Behandlung oft selbst. Dieses Modul zeigt, **welche Probleme dadurch entstehen** und wie man sie mit **Robinson's Transformation** und **Causal Forests** lösen kann.

---

## 1. Zwei Quellen von Verzerrung (Bias)

In Beobachtungsstudien (nicht-experimentellen Daten) gibt es **zwei** separate Probleme:

### a) Regularisierungs-Bias (auch in Experimenten!)

Entsteht, wenn ML-Methoden die Basisfunktionen μ₁(X) und μ₀(X) **unterschiedlich stark** regularisieren.

**Konkretes Beispiel:**
- Es gibt wenige behandelte, aber viele unbehandelte Personen.
- ML passt μ̂₀(X) flexibel an (viele Daten → gute Anpassung).
- ML regularisiert μ̂₁(X) stark (wenig Daten → konservative Schätzung).
- Die Differenz μ̂₁(X) − μ̂₀(X) zeigt dann **Schein-Heterogenität**, die gar nicht existiert!

### b) Confounding-Bias (nur in Beobachtungsstudien)

Entsteht, weil die Behandlungswahrscheinlichkeit mit den Merkmalen X korreliert ist.

**Beispiel:** Patienten mit schlechterem Gesundheitszustand bekommen eher die Behandlung. Dann korreliert der Baseline-Effekt (wie krank jemand ist) mit der Behandlungszuweisung → naive Schätzungen sehen Heterogenität, wo keine ist.

---

## 2. Warum einfache ML-Ansätze versagen

### T-Learner (Two Model Approach)

**Idee:** Schätze μ̂₁(X) und μ̂₀(X) getrennt, bilde die Differenz.

**Problem:** Beide Modelle werden unterschiedlich regularisiert → Regularisierungs-Bias.

### S-Learner (Single Model Approach)

**Idee:** Schätze ein einziges Modell mit Behandlung W als zusätzliche Variable.

**Problem:** Wenn der Behandlungseffekt klein ist, wird W oft „weg-regularisiert" → der Effekt verschwindet.

### X-Learner (Cross-Learner)

**Idee:** Ein verbesserter Ansatz in drei Schritten:

1. Schätze μ̂₀(X) und μ̂₁(X) getrennt
2. Berechne **imputierte Behandlungseffekte**:
   - Für Behandelte: Ỹ(1) = Yᵢ − μ̂₀(Xᵢ) (beobachtetes Y minus geschätztes Kontroll-Outcome)
   - Für Kontrolle: Ỹ(0) = μ̂₁(Xᵢ) − Yᵢ (geschätztes Behandlungs-Outcome minus beobachtetes Y)
3. Sage diese imputierten Effekte als Funktion von X vorher
4. Kombiniere die Vorhersagen, gewichtet nach Propensity Score

**Vorteil:** Reduziert den Regularisierungs-Bias erheblich.  
**Nachteil:** Kann immer noch unter Confounding-Bias leiden.

---

## 3. Robinson's Transformation – Die elegante Lösung

### Das Ziel

Wir wollen den Behandlungseffekt τ schätzen, ohne dass Confounding-Bias die Ergebnisse verzerrt.

### Die Idee: Alles „herausrechnen", was nicht zum Behandlungseffekt gehört

**Schritt 1:** Definiere zwei Hilfsfunktionen:
- **m(X) = E[Y | X]** – der erwartete Outcome, wenn man die Behandlung ignoriert
- **e(X) = P(W = 1 | X)** – der Propensity Score (Behandlungswahrscheinlichkeit)

**Schritt 2:** Residualisiere (= bereinige):
- **Y\* = Y − m̂(X)** – das „bereinigte" Outcome (Einfluss von X entfernt)
- **W\* = W − ê(X)** – die „bereinigte" Behandlung (erwartete Behandlung entfernt)

**Schritt 3:** Einfache Regression von Y\* auf W\*:

$$Y^* = \tau \cdot W^* + \text{Rauschen}$$

Der Koeffizient τ gibt den Behandlungseffekt!

### Warum das so gut funktioniert

| Eigenschaft | Erklärung |
|-------------|-----------|
| **Entfernt Confounding** | Durch Residualisierung wird der Einfluss von X auf Y und W herausgerechnet |
| **Double Robustness** | Fehler in m̂ und ê heben sich in erster Ordnung auf |
| **√n-Konvergenz** | Trotz langsam konvergierender ML-Hilfsschätzungen |
| **Normale Verteilung** | Valide Standardfehler und Konfidenzintervalle |

### Intuition

> **Stell dir vor:** Du willst wissen, ob Dünger Pflanzen größer macht. Aber manche Pflanzen stehen in der Sonne, andere im Schatten – und die in der Sonne bekommen auch mehr Dünger. Wenn du erst den Einfluss der Sonne auf Größe UND auf Düngung herausrechnest, bleibt nur noch der reine Dünger-Effekt übrig.

---

## 4. Robinson für heterogene Effekte (lokal)

Wenn der Behandlungseffekt **nicht konstant** ist, sondern von X abhängt: τ(X), kann man Robinson's Methode **lokal** anwenden:

1. Der Causal Forest liefert adaptive Gewichte α(X) für jede Beobachtung
2. In der „Nachbarschaft" von X führt man eine gewichtete Robinson-Regression durch
3. Ergebnis: τ̂(X) – ein glatter, lokal angepasster Behandlungseffekt

### Kombination mit dem Causal Forest

Der Causal Forest (Modul 04) wird hier erweitert:
- **Splitting:** Maximiere den Unterschied im Behandlungseffekt (geschätzt per Robinson) zwischen linkem und rechtem Knoten
- **Vorhersage:** Nutze Forest-Gewichte + Robinson-Regression
- **Ergebnis:** Robuste τ̂(X)-Schätzung, die sowohl Regularisierungs- als auch Confounding-Bias vermeidet

---

## 5. Implementierung: Cross-Fitting

Wie in Modul 03 ist **Cross-Fitting** entscheidend:

1. Teile Daten in K Folds
2. Für jedes Fold: Schätze m̂(X) und ê(X) auf den **anderen** Folds
3. Berechne Residuen Y\* und W\* mit diesen „fremden" Schätzungen
4. Verwende die Residuen für die Effektschätzung

> **Die Hilfsmodelle werden nie auf denselben Daten evaluiert, auf denen sie geschätzt wurden → keine Overfitting-Rückkopplung.**

---

## 6. Simulation: Warum das alles nötig ist

### Setup mit **konstantem** Behandlungseffekt + Confounding

| Methode | Ergebnis |
|---------|----------|
| **T-Learner (Random Forest)** | „Halluciniert" Heterogenität, die nicht existiert |
| **S-Learner** | Ebenfalls falsche Heterogenität |
| **X-Learner** | Besser, aber nicht perfekt |
| **Causal Forest (mit Robinson)** | ✅ Erkennt korrekt: Effekt ist überall gleich |

> Der Causal Forest mit Robinson's Transformation ist der **einzige Ansatz**, der sowohl Regularisierungs- als auch Confounding-Bias zuverlässig vermeidet.

---

## 7. Wann welche Methode?

| Situation | Empfohlene Methode |
|-----------|-------------------|
| **Experiment**, konstanter Effekt | Robinson (einfachste Variante) |
| **Experiment**, heterogener Effekt | X-Learner oder Causal Forest |
| **Beobachtungsstudie**, konstanter Effekt | Robinson mit Cross-Fitting |
| **Beobachtungsstudie**, heterogener Effekt | Causal Forest mit Robinson ✅ |

---

## Kernbotschaften

1. **Regularisierungs-Bias** entsteht, wenn ML Behandlungs- und Kontrollgruppe unterschiedlich stark regularisiert – auch in Experimenten!
2. **Confounding-Bias** kommt dazu, wenn die Behandlung nicht zufällig ist.
3. **Robinson's Transformation** löst beide Probleme: Erst X-Einfluss herausrechnen, dann den reinen Effekt schätzen.
4. **Double Robustness** garantiert: Selbst wenn m̂ oder ê nicht perfekt sind, liefert die Methode valide Ergebnisse.
5. **Causal Forests mit Robinson** sind der State-of-the-art für heterogene Effekte in Beobachtungsstudien.

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **Beobachtungsstudie** | Studie ohne zufällige Behandlungszuweisung |
| **Regularisierungs-Bias** | Verzerrung durch unterschiedlich starke ML-Regularisierung |
| **Confounding-Bias** | Verzerrung durch nicht-zufällige Behandlungszuweisung |
| **T-Learner** | Getrennte Modelle für Behandelte und Kontrolle |
| **S-Learner** | Ein Modell mit Behandlung als Variable |
| **X-Learner** | Cross-Learner mit imputierten Behandlungseffekten |
| **Robinson's Transformation** | Residualisierung von Y und W bezüglich X |
| **Residualisierung** | X-Einfluss herausrechnen, nur „Bereinigtes" übrig lassen |
| **m(X)** | Erwarteter Outcome, Behandlung ignoriert |
| **e(X)** | Propensity Score – Behandlungswahrscheinlichkeit gegeben X |
| **Cross-Fitting** | Hilfsschätzungen auf anderen Daten als Hauptschätzung |
