# Causal Inference – Kursübersicht

> **Basierend auf dem Stanford-Kurs „Machine Learning & Causal Inference"**  
> Zusammenfassungen auf Deutsch für Einsteiger

---

## Was ist kausale Inferenz?

Kausale Inferenz beschäftigt sich mit der Frage: **Was passiert, wenn ich etwas gezielt verändere?** Das ist etwas anderes als reine Vorhersage oder Korrelation. Während Machine Learning hervorragend darin ist, Muster in Daten zu finden, braucht man für kausale Aussagen zusätzliche Methoden – genau das lernt man in diesem Kurs.

---

## Kursaufbau

Der Kurs ist in sechs aufeinander aufbauende Module gegliedert. Jeder Ordner enthält die Original-Transkripte sowie eine **ausführliche deutschsprachige Zusammenfassung** (`Zusammenfassung.md`).

| Nr. | Modul | Thema | Zusammenfassung |
|-----|-------|-------|-----------------|
| 01 | **Einführung** | Was ist Kausalität? Potenzielle Outcomes, ATE, CATE, Confounder, Counterfactuals | [→ Zusammenfassung](./01%20Intro/Zusammenfassung.md) |
| 02 | **ML in der Ökonometrie** | Supervised Learning, Regularisierung (LASSO, Ridge), Regressionsbäume, Cross-Validation | [→ Zusammenfassung](./02%20ML%20in%20Econometrics/Zusammenfassung.md) |
| 03 | **ML zur Schätzung von ATEs** | Regression Adjustment, Inverse Propensity Weighting, AIPW (Doubly Robust), Cross-Fitting | [→ Zusammenfassung](./03%20ML%20for%20Estimation%20of%20ATEs/Zusammenfassung.md) |
| 04 | **Heterogene Effekte: Causal Trees** | CATE, Causal Trees, Honest Estimation, Causal Forests, adaptive Gewichte | [→ Zusammenfassung](./04%20ML%20for%20CATEs%3A%20Causal%20Trees/Zusammenfassung.md) |
| 05 | **Robuste Schätzung** | Regularisierungs- & Confounding-Bias, T/S/X-Learner, Robinson's Transformation | [→ Zusammenfassung](./05%20Robust%20Estimation%20of%20Treatments/Zusammenfassung.md) |
| 06 | **Verlustfunktionen & Policy Learning** | R-Loss, R-Learner, CATE-Evaluation, QINI-Kurven, Policy Trees | [→ Zusammenfassung](./06%20Loss%20Functions%20for%20Causal%20Inference/Zusammenfassung.md) |

---

## Roter Faden

```
Modul 01          Modul 02           Modul 03
Grundlagen    →   ML-Werkzeuge   →   Durchschnittliche
(Was ist           (LASSO, Bäume,     Behandlungseffekte
 Kausalität?)       Cross-Val)        (ATE mit AIPW)
                                          │
                                          ▼
Modul 06          Modul 05           Modul 04
Loss Functions ←  Robinson's     ←   Heterogene Effekte
& Policy           Transformation     (Causal Trees
  Learning         (Robustheit)        & Forests)
```

1. **Modul 01** legt das Fundament: Was bedeutet Kausalität, warum reicht Korrelation nicht, und welche Denkwerkzeuge braucht man?
2. **Modul 02** führt ML-Methoden ein (Regularisierung, Bäume, Cross-Validation) und grenzt Vorhersage von Kausalität ab.
3. **Modul 03** zeigt, wie man den durchschnittlichen Behandlungseffekt (ATE) mit ML schätzt – von einfacher Regression bis zum doubly-robusten AIPW-Schätzer.
4. **Modul 04** geht einen Schritt weiter: Nicht nur den Durchschnitt, sondern **für wen** eine Behandlung wirkt (CATE), mit Causal Trees und Forests.
5. **Modul 05** adressiert Probleme in Beobachtungsstudien (kein Experiment): Robinson's Transformation entfernt Confounding-Bias.
6. **Modul 06** verbindet alles: Die R-Loss-Funktion macht CATE-Schätzung zum Standard-ML-Problem, und Policy Learning übersetzt Effekte in optimale Entscheidungen.

---

## Wichtige Konzepte auf einen Blick

| Konzept | Modul | Kurzbeschreibung |
|---------|-------|-----------------|
| Potenzielle Outcomes | 01 | Jede Person hat zwei hypothetische Ergebnisse: Y(1) und Y(0) |
| ATE | 01, 03 | Durchschnittlicher Behandlungseffekt über die gesamte Population |
| CATE | 01, 04–06 | Bedingter Behandlungseffekt für Untergruppen |
| Regularisierung | 02 | LASSO/Ridge: Overfitting verhindern durch Bestrafung großer Koeffizienten |
| Cross-Validation | 02, 03 | Datengetriebene Modellwahl durch Kreuzvalidierung |
| AIPW | 03 | Doubly Robust Schätzer: kombiniert Regression + IPW |
| Cross-Fitting | 03, 05 | Hilfsschätzungen auf anderen Daten berechnen als die Hauptschätzung |
| Causal Forest | 04, 05 | Mittelung vieler ehrlicher Causal Trees für glatte CATE-Schätzung |
| Robinson's Transformation | 05, 06 | Residualisierung: X-Einfluss herausrechnen, um den reinen Effekt zu isolieren |
| R-Learner | 06 | Universelles Framework: Minimiere R-Loss mit beliebigem ML |
| Policy Tree | 06 | Entscheidungsbaum als optimale Behandlungsregel |

---

## Für wen ist das?

- **Einsteiger in kausale Inferenz**, die verstehen wollen, wie ML und Kausalität zusammenhängen
- **Datenwissenschaftler**, die über reine Vorhersage hinausgehen wollen
- **Ökonometrie-Studierende**, die moderne ML-Methoden kennenlernen möchten
- **Alle**, die wissen wollen: *„Was passiert, wenn ich etwas verändere?"*

---

## Quelle

Die Zusammenfassungen basieren auf den Vorlesungstranskripten des Kurses **„Machine Learning & Causal Inference"** der Stanford University (Susan Athey & Stefan Wager).