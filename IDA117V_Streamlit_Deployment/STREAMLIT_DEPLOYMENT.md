# Streamlit Community Cloud deployment

Upload this complete project folder as the root of a GitHub repository. Preserve every subfolder, especially 02_Data and 03_Code. The repository includes report, presentation, source tables, analysis outputs and the trusted saved model.

In Streamlit Community Cloud choose Create app, then the repository and main branch. Set the main file path to `03_Code/app.py`. Select Python 3.12 in Advanced settings. Root requirements.txt installs the pinned runtime dependencies. No secrets or paid services are needed.

The app loads the existing saved pipeline. It does not rerun model training on startup. Use Example 1, Example 2 and Example 3 to check deployment, then enter 150 for improved water access in 2011 to check validation.

The final streamlit.app URL is assigned only after successful deployment. A project archive or local interface is not a hosted Streamlit app.
