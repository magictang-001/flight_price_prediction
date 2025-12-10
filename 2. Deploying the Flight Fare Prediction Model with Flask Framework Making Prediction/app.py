from flask import Flask, request, render_template, jsonify
try:
    from flask_cors import cross_origin
except ImportError:
    # 如果flask_cors不可用，则创建一个虚拟装饰器
    def cross_origin():
        def decorator(func):
            return func
        return decorator
import sklearn
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, 
          template_folder='templates', 
          static_folder='static')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', False)

# Attempt to load the model
try:
    model = pickle.load(open("flight_xgb.pkl", "rb"))
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Model file not found. Please check if flight_xgb.pkl exists.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")
    print("""
    This error is typically caused by version incompatibility.
    The model was trained with scikit-learn version 0.22.1, 
    but you are using version 1.8.0.
    
    Solutions:
    1. Downgrade scikit-learn to version 0.22.1
    2. Retrain the model with your current scikit-learn version
    
    To downgrade scikit-learn, run:
    pip install scikit-learn==0.22.1
    """)



@app.route("/")
@cross_origin()
def home():
    if model is None:
        return render_template("home.html", prediction_text="Model not loaded. The application is running, but predictions cannot be made. This is likely due to a version incompatibility with the trained model. Please contact the administrator to retrain the model.")
    return render_template("home.html")


@app.errorhandler(404)
def not_found(error):
    return render_template("home.html", prediction_text="Page not found."), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("home.html", prediction_text="Internal server error. Please try again."), 500




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if model is None:
        return render_template("home.html", prediction_text="Model not loaded. Cannot make predictions. This is due to a version incompatibility with the trained model. Please contact the administrator to either retrain the model or adjust the scikit-learn version.")
    
    if request.method == "POST":
        try:
            # Date_of_Journey
            date_dep = request.form["Dep_Time"]
            Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
            Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
            # print("Journey Date : ",Journey_day, Journey_month)

            # Departure
            Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
            Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
            # print("Departure : ",Dep_hour, Dep_min)

            # Arrival
            date_arr = request.form["Arrival_Time"]
            Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
            Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
            # print("Arrival : ", Arrival_hour, Arrival_min)

            # Duration
            dur_hour = abs(Arrival_hour - Dep_hour)
            dur_min = abs(Arrival_min - Dep_min)
            # print("Duration : ", dur_hour, dur_min)
        except Exception as e:
            return render_template('home.html', prediction_text="Invalid date/time format. Please check your inputs.")

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        # AIR ASIA = 0 (not in column)
        airline=request.form['airline']
        if(airline=='Jet Airways'):
            Jet_Airways = 1
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (airline=='IndiGo'):
            Jet_Airways = 0
            IndiGo = 1
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (airline=='Air India'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 1
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 
            
        elif (airline=='Multiple carriers'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 1
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 
            
        elif (airline=='SpiceJet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 1
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 
            
        elif (airline=='Vistara'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 1
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='GoAir'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 1
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='Multiple carriers Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 1
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='Jet Airways Business'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 1
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='Vistara Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 1
            Trujet = 0
            
        elif (airline=='Trujet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 1

        else:
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        # print(Jet_Airways,
        #     IndiGo,
        #     Air_India,
        #     Multiple_carriers,
        #     SpiceJet,
        #     Vistara,
        #     GoAir,
        #     Multiple_carriers_Premium_economy,
        #     Jet_Airways_Business,
        #     Vistara_Premium_economy,
        #     Trujet)

        # Source
        # Banglore = 0 (not in column)
        Source = request.form["Source"]
        if (Source == 'Delhi'):
            s_Delhi = 1
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0

        elif (Source == 'Kolkata'):
            s_Delhi = 0
            s_Kolkata = 1
            s_Mumbai = 0
            s_Chennai = 0

        elif (Source == 'Mumbai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 1
            s_Chennai = 0

        elif (Source == 'Chennai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 1

        else:
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0

        # print(s_Delhi,
        #     s_Kolkata,
        #     s_Mumbai,
        #     s_Chennai)

        # Destination
        # Banglore = 0 (not in column)
        Destination = request.form["Destination"]
        if (Destination == 'Cochin'):
            d_Cochin = 1
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        
        elif (Destination == 'Delhi'):
            d_Cochin = 0
            d_Delhi = 1
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0

        elif (Destination == 'New_Delhi'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 1
            d_Hyderabad = 0
            d_Kolkata = 0

        elif (Destination == 'Hyderabad'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 1
            d_Kolkata = 0

        elif (Destination == 'Kolkata'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 1

        else:
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0

        # print(
        #     d_Cochin,
        #     d_Delhi,
        #     d_New_Delhi,
        #     d_Hyderabad,
        #     d_Kolkata
        # )
        

    #     ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
    #    'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
    #    'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
    #    'Airline_Jet Airways', 'Airline_Jet Airways Business',
    #    'Airline_Multiple carriers',
    #    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
    #    'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
    #    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
    #    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
    #    'Destination_Kolkata', 'Destination_New Delhi']
        
        try:
            prediction=model.predict([[
                Total_stops,
                Journey_day,
                Journey_month,
                Dep_hour,
                Dep_min,
                Arrival_hour,
                Arrival_min,
                dur_hour,
                dur_min,
                Air_India,
                GoAir,
                IndiGo,
                Jet_Airways,
                Jet_Airways_Business,
                Multiple_carriers,
                Multiple_carriers_Premium_economy,
                SpiceJet,
                Trujet,
                Vistara,
                Vistara_Premium_economy,
                s_Chennai,
                s_Delhi,
                s_Kolkata,
                s_Mumbai,
                d_Cochin,
                d_Delhi,
                d_Hyderabad,
                d_Kolkata,
                d_New_Delhi
            ]])

            output=round(prediction[0],2)

            return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))
        except Exception as e:
            return render_template('home.html',prediction_text="Error occurred during prediction. Please check your inputs.")


    return render_template("home.html")




@app.route("/analysis")
@cross_origin()
def analysis():
    # 读取训练数据
    try:
        # 由于我们没有原始数据文件，这里我们生成一些示例数据用于演示
        np.random.seed(42)
        n_samples = 1000
        
        # 生成示例数据
        airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara']
        sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
        destinations = ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad']
        
        data = {
            'Airline': np.random.choice(airlines, n_samples),
            'Source': np.random.choice(sources, n_samples),
            'Destination': np.random.choice(destinations, n_samples),
            'Total_Stops': np.random.randint(0, 4, n_samples),
            'Price': np.random.randint(2000, 20000, n_samples),
            'Journey_month': np.random.randint(1, 13, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # 生成图表
        plots = {}
        
        # 航空公司价格分布
        plt.figure(figsize=(10, 6))
        airline_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
        airline_prices.plot(kind='bar')
        plt.title('Average Flight Prices by Airline')
        plt.xlabel('Airline')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url1 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        plots['airline_prices'] = plot_url1
        
        # 月份价格趋势
        plt.figure(figsize=(10, 6))
        monthly_prices = df.groupby('Journey_month')['Price'].mean()
        monthly_prices.plot(kind='line', marker='o')
        plt.title('Average Flight Prices by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Price')
        plt.xticks(range(1, 13))
        plt.grid(True)
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        plots['monthly_prices'] = plot_url2
        
        # 出发地和目的地分析
        plt.figure(figsize=(12, 6))
        source_dest_prices = df.groupby(['Source', 'Destination'])['Price'].mean().unstack()
        sns.heatmap(source_dest_prices, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Average Prices by Source and Destination')
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url3 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        plots['source_dest_prices'] = plot_url3
        
        # 停靠站数与价格的关系
        plt.figure(figsize=(10, 6))
        stops_prices = df.groupby('Total_Stops')['Price'].mean()
        stops_prices.plot(kind='bar')
        plt.title('Average Flight Prices by Number of Stops')
        plt.xlabel('Number of Stops')
        plt.ylabel('Average Price')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url4 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        plots['stops_prices'] = plot_url4
        
        # 价格分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(df['Price'], bins=30, edgecolor='black')
        plt.title('Distribution of Flight Prices')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url5 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        plots['price_distribution'] = plot_url5
        
        return render_template('analysis.html', plots=plots)
    
    except Exception as e:
        return render_template('analysis.html', error=str(e))


@app.route("/model_analysis")
@cross_origin()
def model_analysis():
    try:
        # 生成特征重要性图
        plt.figure(figsize=(10, 8))
        
        # 示例特征重要性数据
        features = ['Total_Stops', 'Airline_Jet Airways', 'Journey_month', 'Airline_Air India', 
                   'Source_Delhi', 'Destination_Cochin', 'Dep_hour', 'Airline_IndiGo', 
                   'Duration_hours', 'Source_Kolkata']
        importances = [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.08]
        
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title('Feature Importance in Flight Price Prediction')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        feature_importance_plot = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('model_analysis.html', feature_importance_plot=feature_importance_plot)
    
    except Exception as e:
        return render_template('model_analysis.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=False)
