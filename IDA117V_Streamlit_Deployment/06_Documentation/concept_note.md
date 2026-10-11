# One-page concept note

Problem: Water and sanitation access differ substantially across South African municipalities. Planning teams need transparent evidence to screen municipalities for more detailed service assessments.

Stakeholder: municipal water and sanitation planning teams, with provincial support. Affected population: households within the 213 metropolitan/local municipalities represented by the Census tables.

Task: classify a municipality into higher or lower 2022 access need. Higher need means improved water or sanitation reaches fewer than 90% of households. Inputs exclude the percentages that construct the target. Prediction unit is a municipality. There is no future prediction horizon. This is a retrospective demonstration of screening where current target measurements are unavailable.

Technical success: macro-F1 above the majority-class baseline on untouched test municipalities. Service-delivery screening indicator: recall for higher-need municipalities, complemented by false-negative counts. Real-world improvement in access is not measured by this offline experiment. False negatives miss municipalities that need investigation. False positives spend limited assessment capacity on lower-need municipalities.

Method: split 127 train, 43 validation and 43 test municipalities with seed 42 and stratification. Fit preprocessing only on training partitions. Compare a DummyClassifier with Logistic Regression, Random Forest, a compact MLP and a compact residual tabular neural network. Select by validation macro-F1, optimise the champion using training CV, conduct three controlled ablations and compare paired test correctness with exact McNemar.

Public value: assist human prioritisation of field assessments while preserving measured backlog indicators as the primary funding evidence. Risks include geographic bias, ecological misinterpretation and stale data. Public aggregate data, explicit limitations and subgroup audits reduce these risks. No promised improvement in service provision or statistically significant gain is assumed.
