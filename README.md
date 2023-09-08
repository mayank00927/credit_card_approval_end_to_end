# Credit Card approval prediction

### Problem Statement - Banks were not able to find potential customers for their credit cards. 

### Target Stakeholders - Commercial banks or Third parties who sell credit cards of multiple banks

#### Dataset sources :
csv files are given in notebook/data
  
#### Dataset Information :
There are 18 independent columns in dataset

1. Ind_ID: Client ID
2. Gender: Gender information
3. Car_owner: Having car or not
4. Property_Owner: Having property or not
5. Children: Count of children
6. Annual_Income: Annual income
7. Type_Income: Income type
8. Education: Education level
9. Marital_Status: Marital_status
10. Housing_Type: Living style
11. Birthday_Count: Use backward count from current day (0), -1 means yesterday.
12. Employed_Days: Start date of employment. Use backward count from current day (0). Positive value means, individual is currently unemployed.
13. Mobile_Phone: Any mobile phone
14. Work_Phone: Any work phone
15. Phone: Any phone number
16. Email_Id: Any email ID (Yes-0 or No-1)
17. Type_Occupation: Occupation
18. Family_Members: Family size
    
**Target Column**
* Label: 0 is application approved and 1 is application rejected.

  
  -------------------------------------------------------------------------------------------------------------------------------------------------

#### Model Training Approach - Classification 

## Approach for the project
**EDA :** checked dataset shape and descriptive statistics of data. found null values in "Gender","Annual_Income","Birthday_Count" and "Type_Occupation".
* Noticed imbalanced data as Approval cards "Label" were more as compared to rejected cards "Label"
* Performed univariate and bivariate analysis to get in depth of dataset.

For insight in to data please refer - notebook/EDA/python-notebook file

**Data Ingestion :** Read both csv files and merged those files.

**Data Preprocessing :** Renamed the feature columns properly, corrected conflict values of multiple columns, found out columns which has null values present , feature extraction and data split into train and test.
* Using columntransformer and pipeline - imputed missing values as per mode and median, ordinal encoding and one hot encoding of features then after standard scaling of data and saving training data and test data array in artifacts folder.

**Model Training :** In this phase training of multiple models happens and considering roc score and precision score we find best model and saving model in to pickle file.

**Prediction Pipeline :** 
This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

**Flask App creation :**
Flask app is created with User Interface to predict the Student math score inside a Web Application.

## Scrrenshot of flask app running on localhost :


![Screenshot (34)](https://github.com/mayank00927/credit_card_approval_end_to_end/assets/96683686/2804262b-6807-40dc-b4d2-17570ae49454)
