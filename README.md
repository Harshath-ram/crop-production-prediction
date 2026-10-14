# Crop Production Prediction (India)

This project predicts how much of a crop a district in India will produce, based on the state, district, season, crop, year and the area that was cultivated. I compared Linear Regression, Random Forest, XGBoost and a stacking ensemble of all three.

## Dataset

I used the "Crop Production in India" dataset from Kaggle, which comes from data.gov.in:
https://www.kaggle.com/datasets/abhinand05/crop-production-in-india

It has 246,091 rows covering 1997 to 2015. Each row is one crop in one district for one season, with the area in hectares and the production in tonnes.

## What I did

**Cleaning.** About 3,700 rows had no production value, so I dropped them. Some text columns had extra spaces at the end (for example "Kharif     "), which I stripped. While checking totals per crop, I noticed Coconut was around 20 times bigger than Sugarcane. That turned out to be because coconut is recorded in number of nuts, not tonnes, so I removed it.

**Exploring the data.** Production values are very uneven. Most rows are a few hundred tonnes, but some go into the crores. Plotting log(production) gave a much more normal-looking distribution. The correlation between area and production was only 0.24 on raw values, but it went up to 0.89 after taking logs.

**Features.** I used log of area, year, state, district, season and crop. I also made a combined district + crop column, because the same crop gives very different yields in different districts. The target is log(production).

**Models.** Linear Regression used one-hot encoding. Random Forest and XGBoost used ordinal encoding plus target encoding for the district + crop column, since it has more than 22,000 unique values. I tuned XGBoost with RandomizedSearchCV and then built a stacking ensemble with Ridge as the final model.

**Testing.** First a normal 75/25 train-test split. Then, since the real use would be predicting future seasons, I also trained only on 1997–2010 and tested on 2011–2015.

## Results

| Model | R² (random split) | R² (2011–2015) |
|---|---|---|
| XGBoost | 0.979 | 0.965 |
| Stacking Ensemble | 0.979 | 0.959 |
| Random Forest | 0.971 | 0.952 |
| Linear Regression | 0.944 | 0.954 |

A few things I found interesting:

- My first attempt at Linear Regression on raw tonnes gave an R² of just 0.16. Switching the target to log(production) took it to 0.94 without changing anything else.
- The stacking ensemble and XGBoost were tied on the random split, but XGBoost did better on the future years, so I went with XGBoost as the final model.
- Area and the district + crop feature were by far the most important inputs.

## How to run

pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Keep `Crop Production.csv` in the same folder as the notebook, open `Crop Production Prediction.ipynb` in Jupyter, and run all cells. The full notebook takes around 10–15 minutes, mostly because of the tuning and stacking steps.

## What could be improved

The dataset has no weather or soil information, so the model can't explain why production drops in a drought year. Adding rainfall and temperature data would probably help the most. The data also stops at 2015, and 2015 itself has very few rows. Later I'd like to put the model behind a small Streamlit app.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook
