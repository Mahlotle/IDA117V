# IDA117V: Municipal water and sanitation access needs

This package contains an executed student research prototype using Statistics South Africa's Census 2011/2022 municipal tables. The project owner's approved target is access below 90% for water or sanitation. It is not an official government rating.

## Start in Windows PowerShell

Extract the whole ZIP. Open `IDA117V_Project` in VS Code. In its terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\03_Code\requirements.txt
.\.venv\Scripts\python.exe .\03_Code\run_analysis.py
.\.venv\Scripts\python.exe -m streamlit run .\03_Code\app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). Using the venv Python directly avoids a PowerShell activation-policy change. To open the notebook, choose this venv as the Python/Jupyter kernel in VS Code. For notebook support, install `03_Code/requirements_notebook.txt` with the same venv Python.

The interface can run immediately from the included saved model after installation. Running the analysis again deterministically rebuilds that model and evidence. Exact floating-point equality can vary across CPUs or library builds. Training took 42.0 seconds in the recorded environment; other machines vary.

## Execution order

1. `run_analysis.py` reads original XLSX, builds data, audits, splits, fits all models, searches only the champion, performs ablations and learning curves, then evaluates locked models once per run on the test set.
2. `make_documentation.py` refreshes the data dictionary and documentation.
3. `build_report.py` regenerates the technical report and contribution form.
4. `app.py` loads `final_model.joblib`; it does not train during prediction.
5. `Data_Science_Project.ipynb` can reproduce the analysis and inspect its outputs.

Presentation and report outputs are included. Reference URLs and source terms are in `06_Documentation`. No downloads or hidden manual cleaning steps are needed to run the analysis.

## Recorded results

213 unique municipalities; 127 train / 43 validation / 43 test. Selected family: Residual network. Test accuracy: 0.8372; macro-F1: 0.8110; higher-need recall: 0.8667. Exact McNemar against MLP: p=0.25. The improvement is not statistically significant at alpha=0.05.

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

## Verification

Run `python 03_Code/verify_interface.py` to recheck three valid profiles and one invalid profile through Streamlit AppTest. The saved execution evidence is in `05_Results/interface_ui_tests.json`.
