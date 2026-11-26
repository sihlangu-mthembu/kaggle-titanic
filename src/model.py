"""Titanic survival: feature engineering, model comparison with repeated CV, submission.

Usage: python3 src/model.py <data dir> <out dir>
Group-survival features are computed from the training rows of each fold only, so CV scores
are honest estimates of leaderboard accuracy.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA, OUT = sys.argv[1], sys.argv[2]
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
full = pd.concat([train, test], ignore_index=True)

# --- base features (no target involved) -------------------------------------------------
full["Title"] = full.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
full["Title"] = full.Title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Mrs", "the Countess": "Mrs", "Dona": "Mrs"})
full.loc[~full.Title.isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
full["Surname"] = full.Name.str.split(",").str[0].str.strip()
full["Embarked"] = full.Embarked.fillna("S")
full["Fare"] = full.Fare.fillna(full.groupby("Pclass").Fare.transform("median"))
full["TicketSize"] = full.groupby("Ticket").Ticket.transform("count")
full["FamilySize"] = full.SibSp + full.Parch + 1
full["IsAlone"] = (full.FamilySize == 1).astype(int)
full["FarePP"] = np.log1p(full.Fare / full.TicketSize)
full["Age"] = full.Age.fillna(full.groupby(["Title", "Pclass"]).Age.transform("median"))
full["Deck"] = full.Cabin.str[0].fillna("U").replace({"T": "U"})
full["HasCabin"] = full.Cabin.notna().astype(int)
full["IsFemale"] = (full.Sex == "female").astype(int)
full["Boy"] = (full.Title == "Master").astype(int)
# a travelling group: same surname and fare, or same ticket
full["FamGroup"] = full.Surname + "_" + full.Fare.round(2).astype(str)

ntr = len(train); y = train.Survived.values


def group_feats(df, tr_idx):
    """Survival of the passenger's women-and-children group, from training rows only.
    WCG rule (Chris Deotte's Titanic analysis): women and boys travel as a group; men are
    excluded. A group 'died' if all its training members died, 'lived' if all lived."""
    d = df.copy()
    known = pd.Series(np.nan, index=d.index)
    known.iloc[tr_idx] = y[tr_idx] if len(tr_idx) and tr_idx.max() < ntr else np.nan
    wc = (d.IsFemale == 1) | (d.Boy == 1)
    out = np.zeros((len(d), 3))
    for key in ("FamGroup", "Ticket"):
        g = d[wc].groupby(key).groups
        for _, ix in g.items():
            ix = list(ix)
            if len(ix) < 2:
                continue
            for i in ix:
                others = [j for j in ix if j != i and not np.isnan(known[j])]
                if not others:
                    continue
                r = known[others].mean()
                out[i, 0] = max(out[i, 0], 1.0)  # in a known group
                if r == 0: out[i, 1] = 1.0          # group died
                if r == 1: out[i, 2] = 1.0          # group lived
    return pd.DataFrame(out, columns=["InWCG", "WCGDied", "WCGLived"], index=d.index)


def design(df, tr_idx):
    X = pd.concat([df[["Pclass", "IsFemale", "Age", "SibSp", "Parch", "FamilySize", "IsAlone", "FarePP", "HasCabin", "TicketSize", "Boy"]],
                   pd.get_dummies(df.Title, prefix="T"), pd.get_dummies(df.Embarked, prefix="E"),
                   pd.get_dummies(df.Deck, prefix="D"), group_feats(df, tr_idx)], axis=1).astype(float)
    return X


def wcg_rule(df, idx, G):
    """Gender model, corrected by group survival: boys in surviving groups live, women in
    groups that all died do not."""
    p = df.IsFemale.values[idx].copy()
    p[(df.Boy.values[idx] == 1) & (G.WCGLived.values[idx] == 1)] = 1
    p[(df.IsFemale.values[idx] == 1) & (G.WCGDied.values[idx] == 1)] = 0
    return p


models = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    "rf": lambda: RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=0, n_jobs=-1),
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0),
    "svc": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=0)),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
scores = {k: [] for k in list(models) + ["gender", "wcg_rule", "ensemble"]}
trdf = full.iloc[:ntr].reset_index(drop=True)
for tr, va in cv.split(np.zeros(ntr), y):
    X = design(trdf, tr)
    probs = []
    for name, mk in models.items():
        m = mk().fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[va])[:, 1]
        probs.append(p); scores[name].append(((p > 0.5) == y[va]).mean())
    scores["ensemble"].append(((np.mean(probs, axis=0) > 0.5) == y[va]).mean())
    scores["gender"].append((trdf.IsFemale.values[va] == y[va]).mean())
    G = group_feats(trdf, tr)
    scores["wcg_rule"].append((wcg_rule(trdf, va, G) == y[va]).mean())

summary = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in scores.items()}
for k, (m, s) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:9s} CV accuracy {m:.4f} ± {s:.4f}")
json.dump(summary, open(f"{OUT}/cv.json", "w"), indent=1)

# --- fit on all training rows, predict test ------------------------------------------------
all_tr = np.arange(ntr)
X = design(full, all_tr); Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
best = max(summary, key=lambda k: summary[k][0])
if best == "wcg_rule":
    pred = wcg_rule(full, np.arange(ntr, len(full)), group_feats(full, all_tr))
elif best == "ensemble":
    pred = (np.mean([mk().fit(Xtr, y).predict_proba(Xte)[:, 1] for mk in models.values()], axis=0) > 0.5).astype(int)
elif best == "gender":
    pred = full.IsFemale.values[ntr:]
else:
    pred = models[best]().fit(Xtr, y).predict(Xte)
sub = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": pred.astype(int)})
sub.to_csv(f"{OUT}/submission.csv", index=False)
print("best:", best, "| test predicted survivors:", int(sub.Survived.sum()), "of", len(sub))
