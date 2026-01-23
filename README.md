# 👥 Customer Churn Insights Project | SQL server, RandomForest, Power BI
![image](https://github.com/user-attachments/assets/d1c08730-ca60-42d2-9604-5d0a0366ffa2)

<p align="center">
  <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="40" alt="Python"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/320px-Scikit_learn_logo_small.svg.png" width="60" alt="KMeans"/>
  <img src="https://github.com/user-attachments/assets/e35b27ca-31c4-423f-86d1-1ec31f3df60f" width="40" alt="Custom Image"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/microsoftsqlserver/microsoftsqlserver-plain.svg" width="40" alt="SQL Server"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/cf/New_Power_BI_Logo.svg" width="40" alt="Power BI"/>
</p>


Customer churn is one of the most significant challenges facing modern businesses. As markets become increasingly saturated, understanding why customers leave and more importantly, how to retain them has never been more critical. Churn analysis provides a data-driven approach to tackling this issue by uncovering behavioural trends and service-related factors that influence customer decisions. By integrating predictive modelling and actionable insights, organizations can shift from reactive responses to proactive engagement, ensuring stronger customer relationships and improved business outcomes.

## 🏢 Relevant Sectors and Stakeholders
Understanding customer attrition is vital across industries. This project, initially designed for the telecom sector, provides a scalable framework for churn analysis applicable to finance, retail, and healthcare. By leveraging data analytics and predictive modeling, businesses can proactively address churn, optimize retention strategies, and enhance customer lifetime value.
 
## 📘 Project Overview
This project focuses on **Customer Churn Analysis for a telecom firm.** It utilizes **SQL for** **ETL, Machine Learning (RandomForestClassifier) to identify churn patterns, predict future churners, Power BI for data visualization and provide actionable insights.** While the project is specific to telecom, its methodologies can be applied across retail, finance, and healthcare industries to enhance customer retention strategies.
 
 ## 🎯 Key Objectives
- **Visualizing and Analyzing Customer Data at different levels**:
  -	Demographic (Age, Gender, Marital Status)
  -	Geographic (State-wise churn rates)
  -	Payment & Account Information (Contract Type, Payment Method)
  -	Services Used (Internet, Security, Billing, etc.)
-  **Identifying Churner Profiles** and areas for marketing interventions.
-  **Predicting Customer Churn** using **Machine Learning (RandomForest).**

## 📁 Data Sources
- Data
  - <a href="https://github.com/Shakeel-Data/Churn-prediction-Dashboard/blob/main/Customer%20data.csv">csv</a>
  - <a href="https://github.com/Shakeel-Data/Churn-prediction-Dashboard/blob/main/Prediction%20data.xlsx">xlsx</a>
- SQL
<a href="https://github.com/shakeel-data/churn-prediction-dashboard/blob/main/customer_churn.sql">queries</a>
- Power query
<a href="https://github.com/Shakeel-Data/Churn-prediction-Dashboard/blob/main/Power%20Query%20Transformation%20and%20Measures">DAX</a>
- Python
<a href="https://github.com/shakeel-data/churn-prediction-dashboard/blob/main/customer_churn_predictive_model.ipynb">codes</a>
- Prediction data
<a href="https://github.com/shakeel-data/churn-prediction-dashboard/blob/main/Predictions.csv">csv</a>

# 🪟 Dashboard
![Dashboard overview ](https://github.com/user-attachments/assets/a89377ad-0eb9-424f-8f1b-5aa93121f435)
![Dashboard overview 2](https://github.com/user-attachments/assets/bb63d774-5693-4f5b-9ba1-8eb34347cb4e)

## 🪜 Project Workflow
### 1. 🧩 ETL Process in SQL Server
   -	Created a **database and imported CSV files** into SQL Server using the Import Wizard.
   -	Performed **data exploration** to check distinct values and nulls.
   -	Cleaned data by **removing null values** and inserted it into the **production table**.
   -	Created **views in SQL Server** to integrate with Power BI.

```sql
-- Checking distinct values

SELECT Gender, Count(Gender) as TotalCount,
Count(Gender) * 1.0 / (Select Count(*) from stg_Churn)  as Percentage
from stg_Churn
Group by Gender
```
![image](https://github.com/user-attachments/assets/0ce9338c-4853-450c-8bfd-a92112582e75)

```sql
SELECT Customer_Status, Count(Customer_Status) as TotalCount, Sum(Total_Revenue) as TotalRev,
Sum(Total_Revenue) / (Select sum(Total_Revenue) from stg_Churn) * 100  as RevPercentage
from stg_Churn
Group by Customer_Status
```
![image](https://github.com/user-attachments/assets/5d78791b-a548-479b-addd-1f40a53c5a7e)

```sql
SELECT State, Count(State) as TotalCount,
Count(State) * 1.0 / (Select Count(*) from stg_Churn)  as Percentage
from stg_Churn
Group by State
Order by Percentage desc
```
![image](https://github.com/user-attachments/assets/c36183c3-639a-412a-90c1-80c6396399cc)

```sql
-- Checking nulls

SELECT 
    SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS Customer_ID_Null_Count,
    SUM(CASE WHEN Gender IS NULL THEN 1 ELSE 0 END) AS Gender_Null_Count,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS Age_Null_Count,
    SUM(CASE WHEN Married IS NULL THEN 1 ELSE 0 END) AS Married_Null_Count,
    SUM(CASE WHEN State IS NULL THEN 1 ELSE 0 END) AS State_Null_Count,
    SUM(CASE WHEN Number_of_Referrals IS NULL THEN 1 ELSE 0 END) AS Number_of_Referrals_Null_Count,
    SUM(CASE WHEN Tenure_in_Months IS NULL THEN 1 ELSE 0 END) AS Tenure_in_Months_Null_Count,
    SUM(CASE WHEN Value_Deal IS NULL THEN 1 ELSE 0 END) AS Value_Deal_Null_Count,
    SUM(CASE WHEN Phone_Service IS NULL THEN 1 ELSE 0 END) AS Phone_Service_Null_Count,
    SUM(CASE WHEN Multiple_Lines IS NULL THEN 1 ELSE 0 END) AS Multiple_Lines_Null_Count,
    SUM(CASE WHEN Internet_Service IS NULL THEN 1 ELSE 0 END) AS Internet_Service_Null_Count,
    SUM(CASE WHEN Internet_Type IS NULL THEN 1 ELSE 0 END) AS Internet_Type_Null_Count,
    SUM(CASE WHEN Online_Security IS NULL THEN 1 ELSE 0 END) AS Online_Security_Null_Count,
    SUM(CASE WHEN Online_Backup IS NULL THEN 1 ELSE 0 END) AS Online_Backup_Null_Count,
    SUM(CASE WHEN Device_Protection_Plan IS NULL THEN 1 ELSE 0 END) AS Device_Protection_Plan_Null_Count,
    SUM(CASE WHEN Premium_Support IS NULL THEN 1 ELSE 0 END) AS Premium_Support_Null_Count,
    SUM(CASE WHEN Streaming_TV IS NULL THEN 1 ELSE 0 END) AS Streaming_TV_Null_Count,
    SUM(CASE WHEN Streaming_Movies IS NULL THEN 1 ELSE 0 END) AS Streaming_Movies_Null_Count,
    SUM(CASE WHEN Streaming_Music IS NULL THEN 1 ELSE 0 END) AS Streaming_Music_Null_Count,
    SUM(CASE WHEN Unlimited_Data IS NULL THEN 1 ELSE 0 END) AS Unlimited_Data_Null_Count,
    SUM(CASE WHEN Contract IS NULL THEN 1 ELSE 0 END) AS Contract_Null_Count,
    SUM(CASE WHEN Paperless_Billing IS NULL THEN 1 ELSE 0 END) AS Paperless_Billing_Null_Count,
    SUM(CASE WHEN Payment_Method IS NULL THEN 1 ELSE 0 END) AS Payment_Method_Null_Count,
    SUM(CASE WHEN Monthly_Charge IS NULL THEN 1 ELSE 0 END) AS Monthly_Charge_Null_Count,
    SUM(CASE WHEN Total_Charges IS NULL THEN 1 ELSE 0 END) AS Total_Charges_Null_Count,
    SUM(CASE WHEN Total_Refunds IS NULL THEN 1 ELSE 0 END) AS Total_Refunds_Null_Count,
    SUM(CASE WHEN Total_Extra_Data_Charges IS NULL THEN 1 ELSE 0 END) AS Total_Extra_Data_Charges_Null_Count,
    SUM(CASE WHEN Total_Long_Distance_Charges IS NULL THEN 1 ELSE 0 END) AS Total_Long_Distance_Charges_Null_Count,
    SUM(CASE WHEN Total_Revenue IS NULL THEN 1 ELSE 0 END) AS Total_Revenue_Null_Count,
    SUM(CASE WHEN Customer_Status IS NULL THEN 1 ELSE 0 END) AS Customer_Status_Null_Count,
    SUM(CASE WHEN Churn_Category IS NULL THEN 1 ELSE 0 END) AS Churn_Category_Null_Count,
    SUM(CASE WHEN Churn_Reason IS NULL THEN 1 ELSE 0 END) AS Churn_Reason_Null_Count
FROM stg_Churn;
```

![image](https://github.com/user-attachments/assets/6e4c57fc-fd69-4a5b-979d-6b253a4eeaf0)

```sql
--Monthly Revenue by State

SELECT
     State,
     SUM(Monthly_Charge) AS Monthly_Revenue
FROM stg_Churn
GROUP BY State
ORDER BY Monthly_Revenue DESC;
```
![image](https://github.com/user-attachments/assets/66be6315-7100-4e1a-8b97-5c952322e8ab)

```sql
-- Churned Customers by Payment Method

SELECT 
     Payment_Method,
     COUNT(*) AS Total_Customers,
     SUM(CASE WHEN Customer_Status = 'Churned' THEN 1 ELSE 0 END) AS Churned_Customers
FROM stg_Churn
GROUP BY Payment_Method;
```
![image](https://github.com/user-attachments/assets/96c1a600-e27f-47a2-b97c-7a127c105c74)


```sql
CREATE TABLE prod_Churn

-- Removing null and inserting new data into prod_churn

SELECT 
    Customer_ID,
    Gender,
    Age,
    Married,
    State,
    Number_of_Referrals,
    Tenure_in_Months,
    ISNULL(Value_Deal, 'None') AS Value_Deal,
    Phone_Service,
    ISNULL(Multiple_Lines, 'No') As Multiple_Lines,
    Internet_Service,
    ISNULL(Internet_Type, 'None') AS Internet_Type,
    ISNULL(Online_Security, 'No') AS Online_Security,
    ISNULL(Online_Backup, 'No') AS Online_Backup,
    ISNULL(Device_Protection_Plan, 'No') AS Device_Protection_Plan,
    ISNULL(Premium_Support, 'No') AS Premium_Support,
    ISNULL(Streaming_TV, 'No') AS Streaming_TV,
    ISNULL(Streaming_Movies, 'No') AS Streaming_Movies,
    ISNULL(Streaming_Music, 'No') AS Streaming_Music,
    ISNULL(Unlimited_Data, 'No') AS Unlimited_Data,
    Contract,
    Paperless_Billing,
    Payment_Method,
    Monthly_Charge,
    Total_Charges,
    Total_Refunds,
    Total_Extra_Data_Charges,
    Total_Long_Distance_Charges,
    Total_Revenue,
    Customer_Status,
    ISNULL(Churn_Category, 'Others') AS Churn_Category,
    ISNULL(Churn_Reason , 'Others') AS Churn_Reason 

INTO [db_Churn].[dbo].[prod_Churn]
FROM [db_Churn].[dbo].[stg_Churn];
```

```sql
-- Top 5 States with Highest Churn Rate
SELECT TOP 5
    State,
    COUNT(*) AS Total_Customers,
    SUM(CASE WHEN Customer_Status = 'Churned' THEN 1 ELSE 0 END) AS Churned_Customers,
    ROUND(100.0 * SUM(CASE WHEN Customer_Status = 'Churned' THEN 1 ELSE 0 END) / COUNT(*), 2) AS Churn_Rate
FROM stg_Churn
GROUP BY State
ORDER BY Churn_Rate DESC;
```
![image](https://github.com/user-attachments/assets/228cab1c-6e7b-4940-8910-f2bf773652c8)

```sql
-- Churn Reason Count Breakdown

SELECT 
    Churn_Reason,
    COUNT(*) AS Count
FROM Stg_churn
WHERE Customer_Status = 'Churned'
GROUP BY Churn_Reason
ORDER BY Count DESC;
```
![image](https://github.com/user-attachments/assets/6e6d3dd2-a18d-4cec-8d71-51814085e640)

```sql
-- Creating views for Power BI

Create View vw_ChurnData as
	select * from prod_Churn where Customer_Status In ('Churned', 'Stayed')

Create View vw_JoinData as
	select * from prod_Churn where Customer_Status = 'Joined'
```

### 2. 🗃️ Data Integration
- After completing **data cleaning and exploration** using T-SQL in Microsoft SQL Server, the final dataset was stored as a saved **query/view in the database**.
- This ensured a live, structured connection when **importing the data into Power BI using the native SQL Server connector**, maintaining **data accuracy and consistency** throughout the dashboard development process.

### 3. 📊 Power BI Data Transformation
- Added new calculated columns in **prod_Churn**.
- Created reference tables for:
      -	**Age Group Mapping**
      -	**Tenure Group Mapping**
      -	**Service Categories**

### 4. 🔢 Power BI Measures and Visualization
- Developed **DAX measures** for key performance indicators (KPIs).
- Designed interactive dashboards to **analyze churn patterns** across various segments

### 🔗 power Query Transformations 

```dax
-- Add a new column in prod_Churn
Churn Status = if [Customer_Status] = "Churned" then 1 else 0
```

```dax
-- Change Churn Status data type to numbers
Monthly Charge Range = if [Monthly_Charge] < 20 then "< 20" else if [Monthly_Charge] < 50 then "20-50" else if [Monthly_Charge] < 100 then "50-100" else "> 100"
```

### 田 Creating a new table reference for mapping _age_group

```dax
- Keep only Age column and remove duplicates
Age Group = if [Age] < 20 then "< 20" else if [Age] < 36 then "20 - 35" else if [Age] < 51 then "36 - 50" else "> 50"
```

```dax
AgeGrpSorting = if [Age Group] = "< 20" then 1 else if [Age Group] = "20 - 35" then 2 else if [Age Group] = "36 - 50" then 3 else 4
-- Change data type of AgeGrpSorting
```

### Creating a new table reference for prod_service
- Unpivot services columns
- Rename Column – 
a. Attribute >> Services 
b. Value >> Status


### Summary Page Measures

```dax
Total Customers = Count(prod_Churn[Customer_ID])
```

```dax
New Joiners = CALCULATE(COUNT(prod_Churn[Customer_ID]), prod_Churn[Customer_Status] = "Joined")
```

```dax
Total Churn = SUM(prod_Churn[Churn Status])
```

```dax
Churn Rate = [Total Churn] / [Total Customers]
```

### Churn Prediction -  Page Measures

```dax
Count Predicted Churner = COUNT(Predictions[Customer_ID]) + 0
```

```dax
Title Predicted Churners = "COUNT OF PREDICTED CHURNERS : " & COUNT(Predictions[Customer_ID])
```

### 5. 🌲 Machine Learning (Random Forest)
-	**Data Preparation** for the ML model.
-	Installed necessary **Python libraries**.
-	Imported data and performed **preprocessing**.
-	Trained a **Random Forest Model** to predict churn.
-	Used the model to **predict future churners**.

### Importing Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
```

### Data Load

```python
# pip install openpyxl

# Read the data from the specified sheet into a pandas DataFrame
data = pd.read_excel("C:\yourpath\Prediction_Data.xlsx")

# Display the first few rows of the fetched data
print(data.head())
```
![image](https://github.com/user-attachments/assets/7f4577f2-ea07-4526-a5dc-6359f0fd319b)

### Data Preprocessing
```python
# Drop columns that won't be used for prediction
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)

# List of columns to be label encoded
columns_to_encode = [
    'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',
    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',
    'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
    'Payment_Method'
]
```
```python
# Encode categorical variables except the target variable
label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Manually encode the target variable 'Customer_Status'
data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})
```
```python
# Split data into features and target
X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ▲ Training RandomForestClassifier Model

```python
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)
```

### Evaluate Model
```python
# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/39162bc5-fa01-4357-bdb0-953a40af9cb7)

```python
# Feature Selection using Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(15, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Names')
plt.show()
```
![image](https://github.com/user-attachments/assets/cb650b35-6b56-4061-8d54-0d7af87128e5)

### Use Model for Prediction of New Data
```python
# Define the path to the Joiner Data Excel file
file_path = r"C:\yourpath\Prediction_Data.xlsx"
```
```python
# Define the sheet name to read data from
sheet_name = 'vw_JoinData'
```
```python
# Read the data from the specified sheet into a pandas DataFrame
new_data = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows of the fetched data
print(new_data.head())
```
![image](https://github.com/user-attachments/assets/12f45b3c-0d80-4383-9322-0edf416676a9)

```python
# Retain the original DataFrame to preserve unencoded columns
original_data = new_data.copy()

# Retain the Customer_ID column
customer_ids = new_data['Customer_ID']
```
```python
# Drop columns that won't be used for prediction in the encoded DataFrame
new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)

# Encode categorical variables using the saved label encoders
for column in new_data.select_dtypes(include=['object']).columns:
    new_data[column] = label_encoders[column].transform(new_data[column])
```
### Make Predictions
```python
new_predictions = rf_model.predict(new_data)

# Add predictions to the original DataFrame
original_data['Customer_Status_Predicted'] = new_predictions

# Filter the DataFrame to include only records predicted as "Churned"
original_data = original_data[original_data['Customer_Status_Predicted'] == 1]
```
### Save the results
```python
original_data.to_csv(r"C:\yourpath\Predictions.csv", index=False)
```

### 6. 🪟 Power BI Visualization of Predicted Data
-	Imported **predicted churn data** into SQL Server and Power BI.
-	Created additional **DAX measures** to analyze predicted results.
-	Designed a Churn Prediction Dashboard to visualize potential **churners and customer risk factors**.

## 💼 Business Outcomes
-	**Proactive Customer Retention:** Identifying high-risk customers allows businesses to take preventive measures to reduce churn.
-	**Data-Driven Decision Making:** Visualization and predictive analytics help tailor marketing strategies.
-	**Improved Revenue Management:** Understanding churn trends can enhance customer loyalty programs and pricing models.

## ⚙️ Tools and Technologies
- **Microsoft SQL Server** – Database management and ETL operations
- **Visual Studio Code** – Interactive environment for coding and presenting analysis
- **Python** – Data manipulation, model building, and analysis
  - Libraries: ```pandas```, ```numpy```, ```matplotlib```, ```scikit-learn```, ```joblib``` 
- **Machine Learning Algorithm:** ```RandomForestClassifier ```
- **Microsoft Power BI Desktop** – Data transformation and Visualization 
- **DAX Measures** – Custom calculations in Power BI reports

## 🔚➡️ Conclusion & Next Steps	
This Churn Analysis Dashboard provides a **data-driven foundation for understanding customer attrition and its underlying causes.** The insights reveal critical risk factors related to demographics, service preferences, contract types, and regional trends. Addressing these challenges through proactive retention strategies, service enhancements, and personalized engagement can significantly improve customer loyalty and reduce churn. By integrating **SQL, Power BI, and Machine Learning, it provides a comprehensive solution for customer retention strategies.** The methodology and tools used here can be extended to various industries to drive **data-driven decisions and enhance customer experience.**


### Next Steps:
- Implement **personalized retention** offers for high-risk customers.
- Improve contract structures to encourage **long-term commitments.**
- Deploy **real-time monitoring and alerts for early churn indicators.**


