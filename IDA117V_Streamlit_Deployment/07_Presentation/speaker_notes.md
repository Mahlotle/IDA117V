## Slide 1: Municipal water and sanitation access needs

Suggested speaker: Member 1. Time: 0:45.

We study municipal service access using Statistics South Africa Census tables. Our question is whether historical services and housing indicators can identify municipalities with higher access needs in 2022. This is a small research prototype. Introduce the five group members using their real names. Source: Stats SA, https://www.statssa.gov.za/?p=17836

## Slide 2: Research question and target

Suggested speaker: Member 1. Time: 0:55.

The higher-need class means water or sanitation access was below 90 percent in 2022. Lower need requires both to reach 90 percent. The 90 percent threshold is our approved planning assumption, not an official rating. We focus on access because these tables do not support the intended reliability question adequately. This is retrospective classification rather than forecasting present outages. Source: report and approved project protocol.

## Slide 3: Dataset and quality audit

Suggested speaker: Member 2. Time: 0:55.

We joined official Addendum and Map tables using municipal codes. There are 213 unique municipalities and no missing retained predictor cells. We excluded all 2011 dwelling category counts after they failed reconciliation with household totals. We preserve raw files and record exact URLs, terms and hashes. Stats SA permits independent processing with acknowledgement, with permission required for sale. Source: 05_Results/data_audit.json and Stats SA report copyright page.

## Slide 4: Predictors and split design

Suggested speaker: Member 2. Time: 0:55.

Four historical access indicators are included, alongside housing and municipal context. Direct 2022 target components and municipality names or identifiers are excluded from predictors. We split before preprocessing. Median imputation and scaling, and categorical imputation and one-hot encoding, are learned only from the appropriate training partition. The heldout sample is random, so it cannot demonstrate transfer to unseen provinces. Source: 03_Code/run_analysis.py and split_manifest.csv.

## Slide 5: Five initial models

Suggested speaker: Member 3. Time: 0:55.

A majority-class dummy provides a baseline. Logistic regression and random forest meet the traditional-model requirement. We implement an MLP and a residual tabular neural network in NumPy. The residual model uses a skip connection and is a small educational architecture inspired by tabular neural research. Both neural models use weighted binary cross entropy, Adam, dropout, L2 and early stopping. Source: neural.py and experiment_log.json. Gorishniy et al. 2021, https://arxiv.org/abs/2106.11959

## Slide 6: Champion selection and optimisation

Suggested speaker: Member 3. Time: 0:55.

We chose the residual model using validation macro-F1 before looking at test results. Four-fold cross validation used training data only. The initial CV score for MLP was slightly lower than residual, and validation clearly favoured residual. We searched 16 configurations only for that family. The selected settings improved validation macro-F1 to 0.894. We did not choose a family using these final test results. Source: validation_comparison.csv and optimisation_search.json.

## Slide 7: Held-out model comparison

Suggested speaker: Member 4. Time: 0:55.

After all selection, optimisation and ablations finished, we generated test predictions once per locked model. The final champion reaches macro-F1 0.811 on 43 cases, compared with 0.735 for MLP. The random forest scores 0.801, close to the champion. This small sample makes those differences uncertain. Source: 05_Results/test_metrics.csv.

## Slide 8: Errors and uncertainty

Suggested speaker: Member 4. Time: 0:55.

The confusion matrix has ten correctly identified lower-need cases and 26 correctly identified higher-need cases. Four higher-need municipalities were missed and three lower-need cases were flagged. Relative to the prespecified MLP comparator, the champion alone got three additional cases right. Exact McNemar p is 0.25, so we cannot claim a significant improvement. Spatial dependence and small counts further limit inference. Source: Final_champion_confusion.csv and summary.json.

## Slide 9: Controlled ablations

Suggested speaker: Member 4. Time: 0:45.

We removed one prespecified group at a time and kept all other settings fixed. The historical access group produces the largest validation drop, about 0.181 macro-F1. Geographic and housing information also help the fitted model. These are predictive associations rather than causal effects. We use validation for ablations and leave test outcomes out of feature selection. Source: ablations.csv.

## Slide 10: Local interface demonstration

Suggested speaker: Member 5. Time: 1:30.

Open the Streamlit app before the presentation. Choose Example 1 and submit, then Example 2 and Example 3. Show lower need at 3.5 percent and higher need at 96.6 and 95.4 percent. Explain that probabilities are uncalibrated model scores. Enter 150 in water access for 2011 to show percentage validation. The CSV mode uses the same validation and saved pipeline. These exact flows were tested with Streamlit AppTest. Source: interface_ui_tests.json.

## Slide 11: Responsible use and limitations

Suggested speaker: Member 5. Time: 0:55.

The tool could support exploration and manual review of municipal patterns. Where measured 2022 water and sanitation coverage already exists, compute the rule directly instead of using a prediction. This model cannot assess present reliability or individual households. We use aggregate data but still consider geographic disadvantage and proxy effects. A future study should gather current reliability data, use spatial or province holdouts and assess calibration. Source: report sections 15 to 17.

## Slide 12: Conclusion and reproducibility

Suggested speaker: Member 5. Time: 0:45.

Our prototype classifies the approved historical access target and produces a working local interface. The final score is promising, but the comparison with MLP is not statistically significant. All sources, splits, code, figures and saved results are supplied. The five members must review the work, understand the methods and enter their real contributions and signatures. Close by inviting questions. Source: README.md and report.
