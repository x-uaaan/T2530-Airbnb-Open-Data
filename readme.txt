Airbnb Analytics System - README

Prerequisites
- Python 3.9+ (recommended)
- Jupyter Notebook or JupyterLab for running the notebooks

Required libraries
- pandas (python -m pip install pandas)
- numpy (python -m pip install numpy)
- matplotlib (python -m pip install matplotlib)
- seaborn (python -m pip install seaborn)
- scikit-learn (python -m pip install scikit-learn)
- pyfpgrowth (python -m pip install pyfpgrowth)
- streamlit (python -m pip install streamlit)

Install libraries (pip)
python -m pip install pandas numpy matplotlib seaborn scikit-learn pyfpgrowth streamlit
python -m pip install jupyter

Run sequence (important)
1) Run data cleaning: open and execute `data cleaning.ipynb`
2) Run data mining: open and execute `data mining copy2.ipynb`
   - This generates `decision_tree_model.pkl` and `decision_tree_meta.json`
3) Run the Streamlit app:
   streamlit run app.py

Notes
- `app.py` reads `Airbnb_Open_Data_cleaned.csv` created by the data cleaning step.
- The Smart Rating feature needs the decision tree artifacts from the data mining step.