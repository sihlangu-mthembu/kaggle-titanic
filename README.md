# Titanic: Machine Learning from Disaster

My entry for Kaggle's [Titanic](https://www.kaggle.com/competitions/titanic) Getting Started competition:
predict which of 418 passengers survived.

## Results

| Submission | Approach | Public leaderboard |
|---|---|---|
| [`submissions/submission_wcg_rule.csv`](submissions/submission_wcg_rule.csv) | Women-and-children-group rule | **0.80382** |
| [`submissions/submission_ensemble.csv`](submissions/submission_ensemble.csv) | Average of 4 models on 30 features | 0.79425 |
| Kaggle's `gender_submission.csv` (baseline) | Women survive, men do not | 0.76555 |

Repeated cross-validation (5 folds x 5 repeats, [results-cv.json](results-cv.json)) put every model at
0.836 to 0.846 and the gender baseline at 0.787. The differences between models were smaller than
the noise, so the leaderboard decided: the simpler rule generalised better.

## Approach

**Features** ([src/model.py](src/model.py)): title from the name (Mr, Mrs, Miss, Master, Rare), sex,
class, age filled from the median of the same title and class, family size, travelling alone,
log fare per person (fare divided by the number of people on the ticket), ticket group size, deck
from the cabin, port of embarkation.

**Group survival.** People travelling together (same surname and fare, or same ticket) tended to
live or die together. Women and boys ("Master") form a group; adult men are left out. A group
"lived" if all its known members survived and "died" if all died. These features are computed only
from the training rows of each fold, so cross-validation scores are not inflated.

**The rule that scored best:** start from "women survive, men do not", then
- boys whose group lived are predicted to survive (8 test passengers), and
- women whose group all died are predicted not to (10 test passengers).

This is the women-and-children-group idea from Chris Deotte's Kaggle analysis
"Titanic using Name only [0.81818]" (<https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818>).

**The ensemble:** logistic regression, random forest, gradient boosting and an SVM on all features,
probabilities averaged.

No external passenger lists or known outcomes were used.

## Run it

```bash
pip install pandas numpy scikit-learn
# download train.csv and test.csv from https://www.kaggle.com/competitions/titanic/data into data/
python3 src/model.py data out
```

It prints the cross-validation table and writes `out/submission.csv` (the best model by CV).

## Data

The competition data is not included: it is subject to Kaggle's competition rules. Download it
from the [data page](https://www.kaggle.com/competitions/titanic/data).
