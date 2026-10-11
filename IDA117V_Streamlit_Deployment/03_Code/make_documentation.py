"""Generate documentation from executed analysis outputs."""
import json,sys,platform
from pathlib import Path
import pandas as pd
from prepare_data import ROOT
from run_analysis import FEATURES,NUM

def main():
    doc=ROOT/'06_Documentation';res=ROOT/'05_Results';s=json.loads((res/'summary.json').read_text())
    refs=[
      ('Statistics South Africa (2024)','The state of basic service delivery in South Africa: Analysis of Census 2022 data. Report 03-01-83.','https://www.statssa.gov.za/publications/Report-03-01-83/Report-03-01-832022.pdf'),
      ('Statistics South Africa (2024)','Addendum tables, accompanying Report 03-01-83.','https://www.statssa.gov.za/publications/Report-03-01-83/ADDENDUM%20TABLES.xlsx'),
      ('Statistics South Africa (2024)','Map tables, accompanying Report 03-01-83.','https://www.statssa.gov.za/publications/Report-03-01-83/MAP%20TABLES.xlsx'),
      ('Statistics South Africa (2024)','A decade of progress in service delivery in South Africa.','https://www.statssa.gov.za/?p=17836'),
      ('Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A. (2021)','Revisiting Deep Learning Models for Tabular Data. NeurIPS.','https://arxiv.org/abs/2106.11959'),
      ('scikit-learn developers (n.d.)','Common pitfalls and recommended practices.','https://scikit-learn.org/1.9/common_pitfalls.html'),
      ('SciPy developers (n.d.)','scipy.stats.binomtest documentation.','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html'),
      ('Republic of South Africa (2013)','Protection of Personal Information Act 4 of 2013.','https://www.gov.za/documents/protection-personal-information-act'),
      ('NumPy developers (n.d.)','NumPy documentation.','https://numpy.org/doc/'),
      ('pandas developers (n.d.)','pandas documentation.','https://pandas.pydata.org/docs/'),
      ('Matplotlib developers (n.d.)','Matplotlib documentation.','https://matplotlib.org/stable/'),
      ('Joblib developers (n.d.)','Joblib documentation.','https://joblib.readthedocs.io/en/stable/'),
      ('Streamlit developers (n.d.)','Streamlit documentation.','https://docs.streamlit.io/'),
      ('ReportLab developers (n.d.)','ReportLab documentation.','https://docs.reportlab.com/'),
      ('Python Software Foundation (n.d.)','Python 3 documentation (standard-library XLSX XML reader and notebook runner).','https://docs.python.org/3/')]
    (doc/'references.json').write_text(json.dumps(refs,indent=2))
    (doc/'REFERENCES.md').write_text('# References\n\n'+'\n\n'.join(f'{a}. {b} {u} (accessed 11 October 2026).' for a,b,u in refs))
    meaning={
    'province':('Province code','Map/5.2 and Addendum/5','WC, EC, NC, FS, KZN, NW, GT, MP, LP'),
    'category':('Municipal subcategory','Addendum/5','A, B1, B2, B3, B4'),
    'water_2011':('Households with improved water, 2011','Addendum/5 G','percent'),
    'sanitation_2011':('Households with improved sanitation, 2011','Addendum/5 K','percent'),
    'refuse_2011':('Households with kerbside refuse removal, 2011','Addendum/5 M','percent'),
    'electricity_2011':('Households using electricity for lighting, 2011','Addendum/5 O','percent'),
    'log_households_2022':('Natural logarithm of 1 plus household count, 2022','Derived from Addendum/1 J','log count'),
    'household_growth_pct':('100 × (households 2022 / households 2011 - 1)','Derived from Addendum/1 G,J','percent change'),
    'household_size_2022':('Population / households in 2022','Addendum/1 K','persons per household'),
    'formal_share_2022':('Formal dwellings / sum of 2022 dwelling categories × 100','Addendum/2 J:M','percent'),
    'traditional_share_2022':('Traditional dwellings / sum of 2022 dwelling categories × 100','Addendum/2 J:M','percent'),
    'informal_share_2022':('Informal dwellings / sum of 2022 dwelling categories × 100','Addendum/2 J:M','percent'),
    'urban_share_2022':('Households in urban geography in 2022','Map/5.2 Q','percent'),
    'rdp_share_2022':('RDP/state-subsidised dwellings among eligible denominator','Addendum/3 H','percent'),
    'water_2022':('Improved-water household access, used only to define/audit target','Addendum/5 H','percent'),
    'sanitation_2022':('Improved-sanitation household access, used only to define/audit target','Addendum/5 L','percent'),
    'higher_need':('1 if water_2022 < 90 or sanitation_2022 < 90, otherwise 0','Project operational rule','binary 0/1'),
    'code':('Municipality identifier, never a model feature','Addendum/5 E','text'),
    'municipality':('Municipality name, never a model feature','Addendum/5 F','text')}
    d=pd.read_csv(ROOT/'02_Data/municipal_data.csv');dictionary=[]
    for c in d:
        if c in meaning:definition,source,unit=meaning[c]
        elif c.startswith(('population_','households_','household_size_')):
            definition=c.replace('_',' ');source='Addendum/1';unit='persons per household' if 'size' in c else 'count'
        elif c in ['refuse_2022','electricity_2022']:
            definition='2022 service access, context only';source='Addendum/5';unit='percent'
        elif c=='other_share_2022':definition='Other dwelling share';source='Addendum/2 M divided by J:M total';unit='percent'
        elif c=='rdp_eligible_denominator_2022':definition='Eligible dwelling count for the RDP percentage';source='Addendum/3 F';unit='count'
        else:definition='100 minus corresponding 2022 access percentage';source='Derived from Addendum/5';unit='percent'
        dictionary.append({'column':c,'definition':definition,'source':source,'unit':unit,'role':'predictor' if c in FEATURES else 'target' if c=='higher_need' else 'context / audit only','dtype':str(d[c].dtype)})
    pd.DataFrame(dictionary).to_csv(ROOT/'02_Data/data_dictionary.csv',index=False)
    (doc/'dataset_card.md').write_text('''# Dataset card

Owner and producer: Statistics South Africa. Publication: Report 03-01-83 (31 October 2024), with Addendum Tables and Map Tables. Exact URLs and citations: REFERENCES.md.

Coverage: Census 2011 and Census 2022, all nine provinces. Unit of analysis: municipality. Retained units: 213 category A/B municipalities (8 metros and 205 local municipalities). Excluded: district, provincial and national totals, because these overlap local/metro units. This is a geographic aggregate dataset, not household microdata.

Access conditions: report copyright page permits processing with Stats SA acknowledgement and an explicit independent-processing statement. Sale requires prior Stats SA permission. Do not describe this as Creative Commons data. Original files remain unchanged in 02_Data/raw. Source SHA-256 values are in data_audit.json.

Target: higher_need = 1 when improved-water OR improved-sanitation access in 2022 is below 90%, otherwise 0. This threshold is a student planning choice approved by the project owner, not a government classification. Classes: 149 higher, 64 lower. No missing retained predictors or duplicate codes.

Features: 14 predictors, including province, municipality category, 2011 service access, 2022 housing/urban context and household characteristics. Current target-defining service values, their backlogs, related indices, composite service index, codes, names, and interruption fields are excluded.

Limitations: only 213 observations; spatial dependence; census timing/boundary alignment; ecological inference; historic results cannot establish current conditions. The 2011 dwelling-category sum disagrees with household totals by more than 1% in all 213 units. Those counts are excluded without inventing replacements. RDP percentages have a restricted eligible denominator. Probabilities are not calibrated. The predictors include 2022 context, so the task is contemporaneous classification, not a forecast from 2011.

No individual personal data are present. Geography and rurality may proxy historical disadvantage. Province/category subgroup checks are descriptive because held-out groups are small.
''')
    (doc/'concept_note.md').write_text('''# One-page concept note

Problem: Water and sanitation access differ substantially across South African municipalities. Planning teams need transparent evidence to screen municipalities for more detailed service assessments.

Stakeholder: municipal water and sanitation planning teams, with provincial support. Affected population: households within the 213 metropolitan/local municipalities represented by the Census tables.

Task: classify a municipality into higher or lower 2022 access need. Higher need means improved water or sanitation reaches fewer than 90% of households. Inputs exclude the percentages that construct the target. Prediction unit is a municipality. There is no future prediction horizon. This is a retrospective demonstration of screening where current target measurements are unavailable.

Technical success: macro-F1 above the majority-class baseline on untouched test municipalities. Service-delivery screening indicator: recall for higher-need municipalities, complemented by false-negative counts. Real-world improvement in access is not measured by this offline experiment. False negatives miss municipalities that need investigation. False positives spend limited assessment capacity on lower-need municipalities.

Method: split 127 train, 43 validation and 43 test municipalities with seed 42 and stratification. Fit preprocessing only on training partitions. Compare a DummyClassifier with Logistic Regression, Random Forest, a compact MLP and a compact residual tabular neural network. Select by validation macro-F1, optimise the champion using training CV, conduct three controlled ablations and compare paired test correctness with exact McNemar.

Public value: assist human prioritisation of field assessments while preserving measured backlog indicators as the primary funding evidence. Risks include geographic bias, ecological misinterpretation and stale data. Public aggregate data, explicit limitations and subgroup audits reduce these risks. No promised improvement in service provision or statistically significant gain is assumed.
''')
    criteria=[('1',10,'Problem, stakeholder, data, ethics','Report pp. 1-2; concept_note.md; dataset_card.md'),('2',18,'Audit, five EDA plots, split, pipeline','Report pp. 3-7; data_audit.json; prepare_data.py; run_analysis.py'),('3',22,'Dummy + two traditional + MLP + second neural','Report pp. 8-9; neural.py; validation_comparison.csv'),('4',16,'Metrics, reports, matrices, ROC, learning curves','Report pp. 10-12; 04_Figures; classification_report JSONs'),('5',10,'Champion-only search and locked test evaluation','Report p. 13; optimisation_search.json; summary.json'),('6',10,'Three ablations and paired McNemar','Report pp. 14-15; ablations.csv; paired_test_predictions.csv; mcnemar.json'),('7',7,'Saved pipeline UI, validation, demos, launch','Report p. 16; app.py; predict.py; interface_demo.json; README.md'),('8',7,'Conclusions, reproducibility, presentation, contributions','Report pp. 17-18; presentation; README.md; contribution_statement.pdf')]
    pd.DataFrame([{'criterion':c,'maximum_marks':m,'requirement':r,'evidence':e,'outstanding':'Group must add names, sign actual contributions, review AI policy and perform live/recorded demo.' if c=='8' else 'Completed evidence; no mark guarantee'} for c,m,r,e in criteria]).to_csv(doc/'rubric_compliance.csv',index=False)
    (ROOT/'README.md').write_text(f'''# IDA117V: Municipal water and sanitation access needs

This package contains an executed student research prototype using Statistics South Africa's Census 2011/2022 municipal tables. The project owner's approved target is access below 90% for water or sanitation. It is not an official government rating.

## Start in Windows PowerShell

Extract the whole ZIP. Open `IDA117V_Project` in VS Code. In its terminal:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r .\\03_Code\\requirements.txt
.\\.venv\\Scripts\\python.exe .\\03_Code\\run_analysis.py
.\\.venv\\Scripts\\python.exe -m streamlit run .\\03_Code\\app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). Using the venv Python directly avoids a PowerShell activation-policy change. To open the notebook, choose this venv as the Python/Jupyter kernel in VS Code. Jupyter/ipykernel are optional notebook dependencies.

The interface can run immediately from the included saved model after installation. Running the analysis again deterministically rebuilds that model and evidence. Exact floating-point equality can vary across CPUs or library builds. Training took {s['runtime_seconds']:.1f} seconds in the recorded environment; other machines vary.

## Execution order

1. `run_analysis.py` reads original XLSX, builds data, audits, splits, fits all models, searches only the champion, performs ablations and learning curves, then evaluates locked models once per run on the test set.
2. `make_documentation.py` refreshes the data dictionary and documentation.
3. `build_report.py` regenerates the technical report and contribution form.
4. `app.py` loads `final_model.joblib`; it does not train during prediction.
5. `Data_Science_Project.ipynb` can reproduce the analysis and inspect its outputs.

Presentation and report outputs are included. Reference URLs and source terms are in `06_Documentation`. No downloads or hidden manual cleaning steps are needed to run the analysis.

## Recorded results

213 unique municipalities; 127 train / 43 validation / 43 test. Selected family: {s['champion']}. Test accuracy: {s['final_test']['accuracy']:.4f}; macro-F1: {s['final_test']['macro_f1']:.4f}; higher-need recall: {s['final_test']['recall']:.4f}. Exact McNemar against {s['comparator']}: p={s['mcnemar']['exact_p']:.2f}. The improvement is not statistically significant at alpha=0.05.

## Demonstration

Choose Example 1, submit, then repeat Example 2 and Example 3. These use documented held-out profiles. Change improved-water access in 2011 to 150 and submit to show the error. Use `demo_valid_inputs.csv` for batch input and `demo_invalid_input.csv` for the invalid batch case. Predictions and tested UI outcomes are in `05_Results`.

## Important interpretation limits

The model classifies historical 2022 municipal access. It cannot predict today's outages, assess a household, establish causation, or replace measured backlogs. Only 213 observations are available and spatial dependence limits statistical inference. Model probabilities are uncalibrated. The 90% threshold is a planning assumption, with 85/95% sensitivity counts documented. All 2011 dwelling-category counts are excluded after a reconciliation failure.

## Group completion

Add the five members' names/student numbers and the group/lecturer information to the contribution form. Record only actual contributions and sign it. Review the institutional AI-assistance disclosure rules, read the report and code, and rehearse a 10–12 minute presentation with every member participating. The included suggested speaking allocation is not a claim of work already performed. No signatures or contributions have been fabricated.

## AI assistance

ChatGPT assisted code development, analysis execution, documentation and presentation preparation. The original neural code is an educational NumPy implementation, inspired by the general residual-network approach in Gorishniy et al. (2021), not a reproduction of their full RTDL architecture. Every group member must review and understand the work. Follow the institution's disclosure and assessment rules.

## Data reuse

Stats SA permits independent processing with acknowledgement. Do not sell original/reprocessed data without Stats SA permission. The original files are preserved in `02_Data/raw`. Software libraries retain their own licences. Load only the included trusted joblib model: pickle/joblib files from unknown sources can execute code.
''')
    # Pin the analysis environment. Streamlit was separately installed and tested.
    import scipy,matplotlib,joblib
    req=f'numpy=={s["environment"]["numpy"]}\npandas=={s["environment"]["pandas"]}\nscikit-learn=={s["environment"]["scikit_learn"]}\nscipy=={scipy.__version__}\nmatplotlib=={matplotlib.__version__}\njoblib=={joblib.__version__}\nstreamlit==1.65.0\nreportlab==4.4.9\n'
    (ROOT/'03_Code/requirements.txt').write_text(req)
    (doc/'group_completion.md').write_text('''# Group review and sign-off

The five members must supply their names/student numbers, group number, lecturer and submission details. Fill and sign contribution_statement.pdf with actual work. Signatures require the members' own action.

Suggested future presentation allocation: member 1 problem/data; member 2 audit/EDA; member 3 traditional models; member 4 neural models/overfitting; member 5 statistics/interface/limitations. All five should understand the full pipeline. These are suggestions, not contribution claims.

The package includes a working local UI and three valid/one invalid tested cases. The group must deliver the live or recorded presentation itself. Confirm the accepted second tabular architecture and any institutional AI disclosure requirements with the lecturer if those were not covered by the stated waiver of approval checkpoints.
''')

if __name__=='__main__':main()
