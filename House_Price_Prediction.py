import os, glob, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

class HousePricePrediction:
    def __init__(self, House_List):
        self.Dataset = House_List
        self.Model = LinearRegression()
    
    def Data_Cleaning(self):
        self.City_Mapping = {City: index for index, City in enumerate(self.Dataset["City"].unique())}  
        self.State_Mapping = {State: index for index, State in enumerate(self.Dataset["State"].unique())}

        # Sử dụng .loc để gán giá trị an toàn
        self.Dataset.loc[:, "City"] = self.Dataset["City"].map(self.City_Mapping)
        self.Dataset.loc[:, "State"] = self.Dataset["State"].map(self.State_Mapping)
        
        self.Dataset = self.Dataset.dropna(subset=['Price', 'Bedroom', 'Bathroom', 'Lot Size', 'City', 'State', 'House Size'])
        return self.Dataset
    
    def Inverse_Mapping(self):  
        self.Inverse_City_Mapping = {index: location for location, index in self.City_Mapping.items()}  
        self.Inverse_State_Mapping = {index: location for location, index in self.State_Mapping.items()}     
        
    def Model_Saving(self, Model, Scaler):
        Main_Path = os.getcwd()
        if os.path.exists('Model Results'):
            files = glob.glob(os.path.join('Model Results', '*'))
            for f in files:
                os.remove(f)
        else:
            os.makedirs("Model Results", exist_ok=True) 
        os.chdir(os.path.join(Main_Path, 'Model Results'))        
        joblib.dump(Model,'Price_Prediction_Model.pkl',)
        joblib.dump(Scaler, 'Scaler.pkl')
        os.chdir(Main_Path)

    def Model_Traning(self):
        
        self.Data_Cleaning()
        self.Inverse_Mapping()
        
        self.X = self.Dataset[['Bedroom', 'Bathroom', 'Lot Size', 'City', 'State', 'House Size']]
        self.Y = self.Dataset['Price']
        
        # Chuẩn hóa dữ liệu
        Scaler = StandardScaler()
        self.X = Scaler.fit_transform(self.X)
        
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        
        # Huấn luyện model 
        self.Model.fit(X_train, Y_train)
        
        # Dự đoán trên tập kiểm tra
        Y_Pred_Test = self.Model.predict(X_test)
        
        Test_City = self.Dataset.loc[Y_test.index, 'City']
        Test_State = self.Dataset.loc[Y_test.index, 'State']
        Test_Lot_Size = self.Dataset.loc[Y_test.index, 'Lot Size']
        Test_House_Size = self.Dataset.loc[Y_test.index, 'House Size']
        Test_Bedrooms = self.Dataset.loc[Y_test.index, "Bedroom"]
        Test_Bathrooms = self.Dataset.loc[Y_test.index, "Bathroom"]

        self.Results = pd.DataFrame({
            'State': Test_State,
            'City': Test_City,
            'Lot Size': Test_Lot_Size,
            'House Size': Test_House_Size,
            'Bedrooms': Test_Bedrooms,
            'Bathrooms': Test_Bathrooms,
            'Actual Price': Y_test,            
            'Predicted Price': Y_Pred_Test,
        })
        
        self.Results['City'] = self.Results['City'].map(self.Inverse_City_Mapping)
        self.Results['State'] = self.Results['State'].map(self.Inverse_State_Mapping)
        
        self.Results['Trend Prediction'] = np.where(self.Results['Predicted Price'].diff() > 0, 'Increase', 'Decrease')
        self.Results['Trend Actual'] = np.where(self.Results['Actual Price'].diff() > 0, 'Increase', 'Decrease')
        
        self.Model_Saving(self.Model, Scaler)
        return self.Results.reset_index(drop = True)
    
    def Plot_Predictions(self):
        Pred_Result = self.Model_Traning()
        Predictions = Pred_Result["Predicted Price"]
        Actuals = Pred_Result["Actual Price"]  # Lấy giá trị thực tế tương ứng
        
        plt.figure(figsize=(20, 12))
        plt.style.use('ggplot')
        
        plt.scatter(Actuals, Predictions, edgecolors=(0, 0, 0)) # Sử dụng giá trị thực tế và dự đoán tương ứng
        plt.plot([min(Actuals), max(Actuals)], [min(Actuals), max(Actuals)], 'k--', lw=4) # Đường tham chiếu
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Prices')
        
        Axis = plt.gca() # Lấy trục hiện tại của biểu đồ
        Axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False)) # Định dạng trục Y để hiển thị theo số thực mà không có offset.
        Axis.yaxis.get_major_formatter().set_scientific(False) # Vô hiệu hóa hiển thị theo định dạng khoa học (scientific notation) trên trục Y.
        Axis.ticklabel_format(style='plain', axis='y') # Thiết lập hiển thị trục Y theo kiểu số thực (plain style).
        
        Axis.xaxis.set_major_formatter(ScalarFormatter(useOffset=False)) 
        Axis.xaxis.get_major_formatter().set_scientific(False)
        Axis.ticklabel_format(style='plain', axis='x')
        
        plt.show()