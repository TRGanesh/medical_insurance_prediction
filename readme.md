## Medical Insurance Prediction - Regression project

**Problem Statement:**
ACME Insurance Inc. offers affordable health insurance to thousands of customer all over the United States.They want to know how much expenditure is there for a customer.Based on that they can provide health insurance.

**Goal:**
To create a Machine Learning Model that can estimate the annual medical expenditure for new customers,using information such as age,sex,BMI,children,smoking habits and region of residence.

---

**About Dataset:**  
Dataset contains actual medical charges incurred by over 1300 customers.  
**Columns**:
``` bash
age - Age of the customer 
sex - Gender of the customer
bmi - Body Mass Index of the customer
children - Total number of children to the customer
smoker - Whether the customer is a smoker or not
region - Regin of residence of the customer
charges - Medical expenditure of the customer
```
---
**Dependencies for Exploratory Data Analysis**
<pre>
<b>Pandas</b> - Provides Data Structures like DataFrames, Series for handling Structured data. Useful for performing tasks like Filtering, Sorting, Grouping, Merging
<b>Numpy</b> - Fundamental library for Numerical computing. Provides powerful Array objects and functions for Mathematical Operations. NumPy's Array operations are significantly faster than native Python Lists
<b>Matplotlib</b> - versatile Python library for creating Static, Interactive, and Animated Visualizations. Matplotlib's object-oriented approach allows precise control over plot elements like axes, labels, and annotations
<b>Seaborn</b> - Python Data Visualization Library built on top of Matplotlib
<b>Plotly</b> - Interactive Python library for creating Web-based Visualizations and Dashboards
<b>Statistics</b> - Provides functions for basic Statistical Operations and Calculations
</pre>
---
**Steps Followed:**
1. Data Exploration
2. Dealing with outliers
3. Data Preprocessing
4. Creating Regression models with different algorithms
5. Comparing Metrics
6. Deploying the best model through Streamlit
---
### Data Exploration
Performed the following tasks to explore the dataset:
- Extracted basic Information about Data-Types and identified any NULL Values across all Columns
- Calculated Fundamental Statistics, including Mean, Median, Mode, and Quartiles for all Numerical Columns
- Plotted a HeatMap which shows Correlation between the Numerical Columns
- Plotted Count Plots for Categorical Columns(Sex, Smoker, Region)
- Plotted Distributions for Numerical Columns(Age, BMI, Charges)
- Performed Grouping based on Region Column, to get the area where more Charges had applied
---
### Dealing with Outliers
**Outliers** are data points that Deviate significantly from the rest of the dataset, often lying Far from the Central Tendency Measures like Mean or Median. Techniques such as **Z-Score**, **IQR (Inter-Quartile Range)**, and Visualization Methods like **Box plots** are commonly used to detect and manage outliers in datasets.
- Removed outliers using IQR
  - **IQR = Q3(75th percentile) - Q1(25th percentile)**
  - **Non_Outliers belongs to the Range (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)**
---
**Dependencies for Machine learning from Scikit-Learn**
1. **Train-Test Split :** Data is split into a Training Set (for Model Training) and a Test Set (for Model Evaluation), with a Common Split Ratio such as 70/30 or 80/20.
2. **GridSearchCV :** Automates the Parameter Selection and Cross-Validation, simplifying the Optimization Process
3. **One-Hot Encoder :** Used for Categorical Variable Encoding. Transforms Categorical Features into a Binary Array, with each column representing a Unique category.
4. **StandardScaler :** Standardizes Features by making Mean to 0 & Standard Deviation to 1. Making them Comparable across Different Scales.
5. **Pickle :** Used to save the Machine Learning files, such as StandardScaler, OneHotEncoder..
---
Models used are 
<pre>
<b>Linear Regression</b>
<b>ElasticNet Regression</b>
<b>Stochastic Gradient Descent Regressor</b>
<b>Support Vector Regressor</b>
<b>Random Forest Regressor</b>
<b>K-Neighbors Regressor</b>
</pre>
---
### Data Preprocessing
- Splitted the Dataset into Training & Testing parts
- Encoding the Categorical Features
- Scaling both Training & Testing data
---
### Modelling
- Created Instances of the Machine Learning Models using Scikit-Learn functions
- Fitting the Models(having default Parameters) with Scaled & Transformed X_train & Y_train data
- Created Parameter Grids for Machine Learning Models to pass in GridSearchCV
- Also Fitted GridSearchCV Models with Training Data
- Printed the Best Parameters(which are suitable for the Current Dataset) for each Model
- With those Best Parameters created the Final Models and Compared Regression Metrics

<img width="858" alt="Screenshot 2024-03-30 at 12 02 33 AM" src="https://github.com/TRGanesh/medical_insurance_prediction/assets/117368449/e1a8a984-29ea-4346-a139-dc95138f999d">

- Files such as Scaler, One Hot Encoder, Final Model are Saved using Pickle Module
