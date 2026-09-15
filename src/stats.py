"""Explanatory statistics for the NEISS data: what circumstantial factors
(known before diagnosis) are associated with a severe outcome?

Deliberately excludes diagnosis_group, which is a predictive feature in
model.py but would make the odds ratios here nearly tautological (e.g.
"fracture increases odds of hospitalization" restates the label).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORMULA = "is_severe ~ age_years + C(sex) + C(season) + C(body_part_group)"


def fit_severity_logit(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["is_severe"] = (data["severity"] == "Severe").astype(int)

    model = smf.logit(FORMULA, data=data).fit(disp=False)

    odds_ratio = np.exp(model.params)
    conf_int = np.exp(model.conf_int())

    result = pd.DataFrame({
        "term": model.params.index,
        "odds_ratio": odds_ratio.values,
        "ci_low": conf_int[0].values,
        "ci_high": conf_int[1].values,
        "p_value": model.pvalues.values,
    })
    result = result[result["term"] != "Intercept"].reset_index(drop=True)
    result["term"] = (
        result["term"]
        .str.replace(r"C\((\w+)\)\[T\.(.+)\]", r"\1 = \2", regex=True)
        .str.replace("age_years", "age (+1 year)")
    )
    return result.sort_values("odds_ratio", ascending=False)
