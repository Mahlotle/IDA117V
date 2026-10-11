# Dataset card

Owner and producer: Statistics South Africa. Publication: Report 03-01-83 (31 October 2024), with Addendum Tables and Map Tables. Exact URLs and citations: REFERENCES.md.

Coverage: Census 2011 and Census 2022, all nine provinces. Unit of analysis: municipality. Retained units: 213 category A/B municipalities (8 metros and 205 local municipalities). Excluded: district, provincial and national totals, because these overlap local/metro units. This is a geographic aggregate dataset, not household microdata.

Access conditions: report copyright page permits processing with Stats SA acknowledgement and an explicit independent-processing statement. Sale requires prior Stats SA permission. Do not describe this as Creative Commons data. Original files remain unchanged in 02_Data/raw. Source SHA-256 values are in data_audit.json.

Target: higher_need = 1 when improved-water OR improved-sanitation access in 2022 is below 90%, otherwise 0. This threshold is a student planning choice approved by the project owner, not a government classification. Classes: 149 higher, 64 lower. No missing retained predictors or duplicate codes.

Features: 14 predictors, including province, municipality category, 2011 service access, 2022 housing/urban context and household characteristics. Current target-defining service values, their backlogs, related indices, composite service index, codes, names, and interruption fields are excluded.

Limitations: only 213 observations; spatial dependence; census timing/boundary alignment; ecological inference; historic results cannot establish current conditions. The 2011 dwelling-category sum disagrees with household totals by more than 1% in all 213 units. Those counts are excluded without inventing replacements. RDP percentages have a restricted eligible denominator. Probabilities are not calibrated. The predictors include 2022 context, so the task is contemporaneous classification, not a forecast from 2011.

No individual personal data are present. Geography and rurality may proxy historical disadvantage. Province/category subgroup checks are descriptive because held-out groups are small.
