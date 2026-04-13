# 02 – Machine Learning in der Ökonometrie

## Worum geht es?

Dieses Modul erklärt, **was Machine Learning (ML) ist**, wie es sich von klassischer Ökonometrie unterscheidet und warum beides sinnvoll kombiniert werden kann. Die Kernfrage: *Kann man ML-Werkzeuge für die empirische Wirtschaftsforschung nutzen – und wo liegen die Grenzen?*

---

## 1. Zwei Arten von Machine Learning

| Typ | Beschreibung | Beispiel |
|-----|-------------|---------|
| **Unsupervised Learning** | Muster finden ohne vorgegebene Labels | Bilder automatisch in Gruppen sortieren (Katzen, Hunde, …) |
| **Supervised Learning** | Aus gelabelten Daten lernen, um Vorhersagen zu treffen | Aus Hauspreisen + Merkmalen lernen, neue Preise vorherzusagen |

> Dieses Modul konzentriert sich auf **Supervised Learning** – das dem Regressionsansatz der Ökonometrie am meisten ähnelt.

---

## 2. Was haben Regression und ML gemeinsam?

Beide versuchen, die Beziehung zwischen **X** (erklärende Variablen) und **Y** (Ergebnis) zu modellieren:

- **Ökonometrie:** Y = f(X) → meist lineare Regression
- **Machine Learning:** Y = f(X) → oft flexiblere Funktionen (neuronale Netze, Random Forests, …)

In ML-Kursen lernt man als erstes: Regression. In Ökonometrie-Kursen lernt man als erstes: Regression. **Es sieht gleich aus – aber die Ziele sind verschieden!**

---

## 3. Der entscheidende Unterschied: Vorhersage vs. Kausalität

### Machine Learning fragt:
> „Kann ich Y möglichst gut vorhersagen, wenn ich X kenne?"

### Ökonometrie fragt:
> „Was passiert mit Y, wenn ich X gezielt verändere?"

### Das McKinsey-Beispiel (Kundenabwanderung)

- **ML-Modell:** Sagt sehr gut vorher, *wer* den Dienst verlassen wird (Churn-Prediction).
- **Problem:** Manche Kunden kündigen, weil sie umziehen – die anzurufen bringt nichts.
- **Kausale Frage:** *Für wen* würde ein Anruf tatsächlich etwas ändern? → Das ist eine **andere Frage**, die ein anderes Modell braucht.

> **Vorhersagen, wer kündigen wird ≠ herausfinden, wen man anrufen sollte.**

---

## 4. Wie Machine Learning funktioniert – das Geheimrezept

### Drei Zutaten

1. **Flexible Funktionen** – ML nutzt komplexere Modelle als lineare Regression (z. B. neuronale Netze, Bäume), die auch nichtlineare Zusammenhänge erfassen können.

2. **Regularisierung** – Die Flexibilität wird bewusst eingeschränkt, um **Overfitting** zu vermeiden (d. h. das Modell lernt nicht nur das Signal, sondern auch das Rauschen in den Daten).

3. **Tuning per Cross-Validation** – Das Ausmaß der Einschränkung wird datengetrieben bestimmt.

### Approximation vs. Overfitting – der zentrale Trade-off

| Modell zu einfach | Modell zu komplex |
|---|---|
| Erfasst den wahren Zusammenhang nicht genau genug | Passt perfekt auf die Trainingsdaten, versagt aber bei neuen Daten |
| → **Approximationsfehler** (Underfitting) | → **Overfitting** |

**Ziel:** Den Sweet Spot finden, wo das Modell komplex genug ist, um den Zusammenhang zu erfassen, aber nicht so komplex, dass es Rauschen mitlernt.

---

## 5. Regularisierung am Beispiel Lineare Regression

### OLS (gewöhnliche Regression ohne Regularisierung)
- Sucht die Koeffizienten, die den Fehler **in den Trainingsdaten** minimieren.
- Kann bei vielen Variablen zu Overfitting führen.

### LASSO-Regression (L1-Regularisierung)
- **Bestraft** große Koeffizienten durch einen Aufschlag auf die Summe der **Absolutwerte**.
- **Effekt:** Setzt viele Koeffizienten exakt auf null → wählt automatisch wichtige Variablen aus.
- Ist **„kapitalistisch"**: Wenn mehrere Variablen ähnlich nützlich sind, wählt sie einen „Gewinner" und setzt die anderen auf null.

### Ridge-Regression (L2-Regularisierung)
- **Bestraft** große Koeffizienten durch einen Aufschlag auf die Summe der **Quadrate**.
- **Effekt:** Schrumpft alle Koeffizienten, setzt aber keinen exakt auf null.
- Ist **„sozialistisch"**: Verteilt die Vorhersagekraft gleichmäßig auf ähnliche Variablen.

### Elastic Net
- **Mischung** aus LASSO und Ridge (zwei Stellschrauben: wie stark regularisieren + wie mischen).

---

## 6. Regressionsbäume (Regression Trees)

Ein Baum teilt den Datenraum in Bereiche auf, indem er nacheinander fragt: *Ist Variable X über oder unter einem Schwellenwert?*

### Beispiel: Hauspreis vorhersagen
```
Badezimmer < 1,5?
├── Ja → Vorhersage: 9,4 (log)
└── Nein → Bodenart = Typ 4,5,6?
    ├── Ja → Vorhersage: 10,2 (log)
    └── Nein → Vorhersage: 11,0 (log)
```

### Vorteil gegenüber linearer Regression
- Kann automatisch **Interaktionseffekte** finden (z. B. „Mehr Zimmer helfen nur bei wenigen Bewohnern")
- Keine Vorab-Annahme über die Form des Zusammenhangs nötig

### Nachteil
- Zu viele Ebenen → Overfitting (jedes Blatt enthält zu wenige Datenpunkte)
- Deshalb: **Regularisierung** durch Begrenzung der Baumtiefe oder Mindestanzahl je Blatt

---

## 7. Cross-Validation – Wie man das richtige Modell wählt

**Problem:** Wie wählt man die richtige Stärke der Regularisierung?

**Lösung: Cross-Validation (Kreuzvalidierung)**

1. Teile die Trainingsdaten in $k$ gleiche Teile (Folds).
2. Halte jeweils einen Teil zurück als „Test".
3. Trainiere das Modell auf den restlichen Teilen.
4. Miss, wie gut das Modell auf dem zurückgehaltenen Teil vorhersagt.
5. Wiederhole für alle Teile und mittele die Fehler.
6. Wähle den Regularisierungsparameter, der den kleinsten Durchschnittsfehler ergibt.

> **Analogie:** Wie Probeklausuren schreiben, um sich optimal auf die echte Klausur vorzubereiten.

---

## 8. Die Firewall – Trainings- vs. Testdaten

| Datenteil | Zweck |
|-----------|-------|
| **Fitting Sample** (Trainingsdaten) | Modell auswählen, Regularisierung tunen, Funktion schätzen |
| **Hold-out Sample** (Testdaten) | Endgültige Bewertung: Wie gut funktioniert das Modell wirklich? |

> **Wichtig:** Die Testdaten dürfen **nie** beim Training benutzt werden – das ist die „Firewall". Nur so bekommt man eine ehrliche Einschätzung der Modellqualität.

---

## 9. Warum ML nicht automatisch kausale Aussagen liefert

### LASSO-Experiment: Instabile Variablenauswahl

Wenn man den LASSO 10-mal auf verschiedene zufällige Stichproben derselben Daten anwendet:
- Manche Variablen werden **immer** ausgewählt
- Viele Variablen werden **mal ja, mal nein** ausgewählt
- Die **Vorhersagequalität** bleibt trotzdem fast gleich!

> In hochdimensionalen Daten gibt es viele Variablen, die sich gegenseitig ersetzen können. Welche der LASSO wählt, ist teilweise Zufall.

### Drei Arten von Verzerrung (Bias)

| Bias-Typ | Erklärung |
|----------|-----------|
| **Kompaktifizierungs-Bias** | LASSO lässt wichtige Variablen weg, weil andere ähnlich genug sind |
| **Expansions-Bias** | Eine eigentlich unwichtige Variable wird einbezogen, weil sie zufällig gut korreliert |
| **Shrinkage-Bias** | Alle Koeffizienten werden Richtung null geschrumpft |

→ Deshalb: **ML-Koeffizienten nicht kausal interpretieren!**

---

## 10. Wann ML sinnvoll einsetzen?

### ML ist **super** für:
- **Datenvorverarbeitung**: Textdaten, Bilddaten, Satellitendaten in nutzbare Variablen umwandeln
- **Echte Vorhersageprobleme** (Prediction Policy Problems): z. B. Armutsbekämpfung – welche Haushalte brauchen Hilfe?
- **Kontrollvariablen** in kausalen Modellen: ML zur Schätzung von Propensity Scores oder Baseline-Effekten nutzen

### ML kann **nicht** (allein):
- Kausale Effekte schätzen
- Sagen, welche Variablen *wirklich ursächlich* für ein Ergebnis sind
- Die Frage beantworten: „Was passiert, wenn ich etwas verändere?"

---

## 11. Was die Ökonometrie von ML lernen kann (und umgekehrt)

| Ökonometrie lernt von ML | ML lernt von Ökonometrie |
|--------------------------|--------------------------|
| Datengetriebene Modellwahl statt Handarbeit | Zwischen Vorhersage und Kausalität unterscheiden |
| Systematische Robustheitsprüfung | Identifikationsprobleme ernst nehmen |
| Flexible Funktionsformen nutzen | Standard Errors und Inferenz beachten |
| Validierung mit Testdaten | Hold-out-Sets klug konstruieren |

---

## Kernbotschaften

1. **ML = fantastisch für Vorhersagen**, aber Vorhersagen ≠ kausale Erkenntnisse.
2. **Regularisierung + Cross-Validation** sind das Geheimrezept, um Overfitting zu vermeiden.
3. **ML-Koeffizienten sind instabil und verzerrt** – man sollte sie nicht wie kausale Parameter interpretieren.
4. Die **Kombination aus ML und kausaler Inferenz** bietet großes Potenzial (→ folgende Module).

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **Supervised Learning** | Lernen aus gelabelten Daten (Y + X → Modell) |
| **Unsupervised Learning** | Muster finden ohne Labels |
| **Overfitting** | Das Modell lernt Rauschen statt Muster |
| **Regularisierung** | Bewusste Einschränkung der Modellkomplexität |
| **LASSO** | L1-Regularisierung; setzt Koeffizienten auf null |
| **Ridge** | L2-Regularisierung; schrumpft alle Koeffizienten gleichmäßig |
| **Cross-Validation** | Datengetriebene Wahl des Regularisierungsparameters |
| **Bias-Varianz-Trade-off** | Zu einfach = Bias; zu komplex = Varianz; Ziel = Balance |
| **Regressionsbäume** | Vorhersage durch rekursive Ja/Nein-Fragen auf Variablen |
| **Hold-out Sample** | Zurückgehaltene Testdaten für ehrliche Modellbewertung |
| **Propensity Score** | Wahrscheinlichkeit, behandelt zu werden, gegeben X |
