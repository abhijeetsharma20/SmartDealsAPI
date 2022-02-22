from app import predict
import uvicorn
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the Random Forest Classifier Bike model
filename1 = "Bike_Price_ML_Model_main.pkl"
classifier1 = pickle.load(open(filename1, "rb"))

# Load the Random Forest Classifier Car model
filename2 = "Car_Price_ML_Model.pkl"
classifier2 = pickle.load(open(filename2, "rb"))

class BikeInput(BaseModel):
    seller_type : int
    owner : int 
    km_driven : int 
    ex_showroom_price : float
    brand : int 
    no_of_yr : int 

class CarInput(BaseModel):
    Present_Price : float
    Kms_Driven : int 
    Owner : int 
    Year : int 
    Fuel_Type_Diesel : int 
    Fuel_Type_Petrol : int 
    Seller_Type_Individual : int 
    Transmission_Mannual : int 

app = FastAPI()

@app.post('/bike_prediction')
def get_bike_price(data: BikeInput):
    received = data.dict()

    seller_type = received['seller_type']
    owner = received['owner']
    km_driven = received['km_driven']
    ex_showroom_price = received['ex_showroom_price']
    brand = received['brand']
    no_of_yr = received['no_of_yr']

    input = np.array(
                [[seller_type, owner, km_driven, ex_showroom_price, brand, no_of_yr]]
            )
    my_prediction = classifier1.predict(input).tolist()[0]
    return {'prediction': my_prediction}

@app.post('/car_prediction')
def get_car_price(data : CarInput):
    received = data.dict()

    Present_Price = received['Present_Price']
    Kms_Driven = received['Kms_Driven']
    Owner = received['Owner'] 
    Year = received['Year'] 
    Fuel_Type_Diesel = received['Fuel_Type_Diesel'] 
    Fuel_Type_Petrol = received['Fuel_Type_Petrol']
    Seller_Type_Individual = received['Seller_Type_Individual'] 
    Transmission_Mannual = received['Transmission_Mannual'] 

    prediction = classifier2.predict(
                [
                    [
                        Present_Price,
                        Kms_Driven,
                        Owner,
                        Year,
                        Fuel_Type_Diesel,
                        Fuel_Type_Petrol,
                        Seller_Type_Individual,
                        Transmission_Mannual,
                    ]
                ]
            ).tolist()[0]

    return {'prediction': prediction}

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=5000, debug=True)