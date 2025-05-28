import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("car_data.csv")

# car name label encoding
car_name_le = LabelEncoder()
dataset["Car_Name"] = car_name_le.fit_transform(dataset["Car_Name"])

# fuel type label encoding
fuel_type_le = LabelEncoder()
dataset["Fuel_Type"] = fuel_type_le.fit_transform(dataset["Fuel_Type"]) 

# seller type label encoding    
seller_type_le = LabelEncoder()
dataset["Seller_Type"] = seller_type_le.fit_transform(dataset["Seller_Type"])

# transmission type label encoding    
transmission_type_le = LabelEncoder()
dataset["Transmission"] = transmission_type_le.fit_transform(dataset["Transmission"])

input_data = dataset.iloc[:,:-1]
output_data = dataset["Selling_Price"]

ss = StandardScaler()
input_data = pd.DataFrame(ss.fit_transform(input_data),columns=input_data.columns)

x_train,x_test,y_train,y_test = train_test_split(input_data,output_data,test_size=0.2,random_state=42)


rf = RandomForestRegressor(n_estimators=100)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)*100 , rf.score(x_train,y_train)*100


# user input prediction
new_data = pd.DataFrame([["dzire",2014,8.06,45780,'Diesel','Dealer','Manual',0]],columns=x_train.columns)

new_data["Car_Name"] = car_name_le.transform(new_data["Car_Name"])
new_data["Fuel_Type"] = fuel_type_le.transform(new_data["Fuel_Type"])
new_data["Seller_Type"] = seller_type_le.transform(new_data["Seller_Type"])
new_data["Transmission"] = transmission_type_le.transform(new_data["Transmission"]) 

new_data = pd.DataFrame(ss.transform(new_data),columns=new_data.columns)

print(rf.predict(new_data))