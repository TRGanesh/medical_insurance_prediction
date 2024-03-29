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
1. **Train-Test Split :** Data is split into a Training Set (for Model Training) and a Test Set (for Model Evaluation), with a Common Split Ratio such as 70/30 or 80/20
2. **GridSearchCV :** Automates the Parameter Selection and Cross-Validation, simplifying the Optimization Process
