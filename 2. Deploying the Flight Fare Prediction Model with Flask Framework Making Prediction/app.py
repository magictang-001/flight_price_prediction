from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
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
from datetime import datetime
from models import db, Flight, Prediction, User
from config import config
from werkzeug.security import generate_password_hash, check_password_hash
warnings.filterwarnings('ignore')

app = Flask(__name__, 
          template_folder='templates', 
          static_folder='static')

# Load configuration
app.config.from_object(config['development'])

# Initialize database
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()

# 设置密钥
app.secret_key = 'your-secret-key-here'

# 添加语言状态管理
app.config['DEFAULT_LANGUAGE'] = 'zh'  # 默认语言为中文

# 登录验证装饰器
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# 管理员权限装饰器
def admin_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to login first', 'error')
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin():
            flash('Access denied. Administrator privileges required.', 'error')
            return redirect(url_for('list_flights'))
        
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

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

# Load price statistics for the gauge chart
price_stats = {
    'min': 1759,
    'max': 79512,
    'mean': 9087,
    '25%': 5277,
    '50%': 8372,
    '75%': 12373
}

try:
    # Try to load from actual data if available
    # The data file is in the sibling directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, '1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices', 'Data_Train.xlsx')
    
    if os.path.exists(data_path):
        # We need openpyxl or xlrd to read excel
        try:
            df_train = pd.read_excel(data_path)
            price_stats = {
                'min': float(df_train['Price'].min()),
                'max': float(df_train['Price'].max()),
                'mean': float(df_train['Price'].mean()),
                '25%': float(df_train['Price'].quantile(0.25)),
                '50%': float(df_train['Price'].quantile(0.50)),
                '75%': float(df_train['Price'].quantile(0.75))
            }
            print("Price statistics loaded from Data_Train.xlsx")
        except ImportError:
             print("Missing optional dependency (openpyxl/xlrd) to read Excel file. Using default stats.")
except Exception as e:
    print(f"Could not load price statistics from file, using defaults: {e}")


@app.route("/")
@cross_origin()
def home():
    if model is None:
        return render_template("home.html", prediction_text="Model not loaded. The application is running, but predictions cannot be made. This is likely due to a version incompatibility with the trained model. Please contact the administrator to retrain the model.")
    return render_template("home.html")


@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 查找用户
        user = User.query.filter_by(username=username).first()
        
        # 验证用户和密码
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            flash('Login successful!', 'success')
            return redirect(url_for('list_flights'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@cross_origin()
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('role', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
@cross_origin()
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # 检查密码是否匹配
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        # 检查邮箱是否已存在
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        # 创建新用户（默认为普通用户）
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password, role='user')
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html')


@app.errorhandler(404)
def not_found(error):
    return render_template("home.html", prediction_text="Page not found."), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("home.html", prediction_text="Internal server error. Please try again."), 500


@app.route('/flights')
@login_required
@cross_origin()
def list_flights():
    """List flights - admins see all flights, users see only their own"""
    try:
        # 检查用户角色
        user = User.query.get(session['user_id'])
        
        # Base query
        query = Flight.query
        
        if not user.is_admin():
            # 普通用户只能看到自己的航班
            query = query.filter_by(user_id=session['user_id'])
            
        # Get distinct values for dropdowns
        distinct_airlines = [r.airline for r in query.with_entities(Flight.airline).distinct().order_by(Flight.airline).all()]
        distinct_sources = [r.source for r in query.with_entities(Flight.source).distinct().order_by(Flight.source).all()]
        distinct_destinations = [r.destination for r in query.with_entities(Flight.destination).distinct().order_by(Flight.destination).all()]
            
        # Filtering
        airline = request.args.get('airline')
        if airline:
            query = query.filter(Flight.airline.ilike(f"%{airline}%"))
            
        source = request.args.get('source')
        if source:
            query = query.filter(Flight.source.ilike(f"%{source}%"))
            
        destination = request.args.get('destination')
        if destination:
            query = query.filter(Flight.destination.ilike(f"%{destination}%"))
            
        departure_time = request.args.get('departure_time')
        if departure_time:
            # Assumes YYYY-MM-DD input
            query = query.filter(db.func.date(Flight.departure_time) == departure_time)
            
        arrival_time = request.args.get('arrival_time')
        if arrival_time:
            query = query.filter(db.func.date(Flight.arrival_time) == arrival_time)
            
        total_stops = request.args.get('total_stops')
        if total_stops is not None and total_stops != '':
            query = query.filter(Flight.total_stops == int(total_stops))

        # Sorting
        sort_by = request.args.get('sort_by')
        sort_order = request.args.get('sort_order', 'asc')
        
        if sort_by == 'price':
            if sort_order == 'desc':
                query = query.order_by(Flight.price.desc())
            else:
                query = query.order_by(Flight.price.asc())
        else:
            # Default sort by ID desc (newest first)
            query = query.order_by(Flight.id.desc())

        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = 10
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # Prepare params for pagination links (remove 'page' to avoid conflict)
        params = request.args.to_dict()
        if 'page' in params:
            del params['page']
            
        return render_template('flights.html', 
                             flights=pagination.items, 
                             pagination=pagination, 
                             user=user,
                             args=request.args,
                             params=params,
                             airlines=distinct_airlines,
                             sources=distinct_sources,
                             destinations=distinct_destinations)
    except Exception as e:
        return render_template('flights.html', error=str(e), flights=[], user=User.query.get(session['user_id']))


@app.route('/flights/add', methods=['GET', 'POST'])
@login_required
@cross_origin()
def add_flight():
    """Add a new flight"""
    if request.method == 'POST':
        try:
            # Get form data
            airline = request.form.get('airline')
            source = request.form.get('source')
            destination = request.form.get('destination')
            departure_time = datetime.strptime(request.form.get('departure_time'), '%Y-%m-%dT%H:%M')
            arrival_time = datetime.strptime(request.form.get('arrival_time'), '%Y-%m-%dT%H:%M')
            total_stops = int(request.form.get('total_stops'))
            price = float(request.form.get('price')) if request.form.get('price') else None
            
            # Create new flight
            flight = Flight(
                airline=airline,
                source=source,
                destination=destination,
                departure_time=departure_time,
                arrival_time=arrival_time,
                total_stops=total_stops,
                price=price,
                user_id=session['user_id']  # 关联到当前用户
            )
            
            # Add to database
            db.session.add(flight)
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Flight added successfully!'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Error adding flight: {str(e)}'}), 400
    
    return render_template('add_flight.html')


@app.route('/flights/edit/<int:id>', methods=['GET', 'POST'])
@login_required
@cross_origin()
def edit_flight(id):
    """Edit an existing flight"""
    # 获取航班信息
    flight = Flight.query.get_or_404(id)
    
    # 检查权限：管理员可以编辑任何航班，普通用户只能编辑自己的航班
    user = User.query.get(session['user_id'])
    if not user.is_admin() and flight.user_id != session['user_id']:
        flash('Access denied. You can only edit your own flights.', 'error')
        return redirect(url_for('list_flights'))
    
    if request.method == 'POST':
        try:
            # Update flight data
            flight.airline = request.form.get('airline')
            flight.source = request.form.get('source')
            flight.destination = request.form.get('destination')
            flight.departure_time = datetime.strptime(request.form.get('departure_time'), '%Y-%m-%dT%H:%M')
            flight.arrival_time = datetime.strptime(request.form.get('arrival_time'), '%Y-%m-%dT%H:%M')
            flight.total_stops = int(request.form.get('total_stops'))
            flight.price = float(request.form.get('price')) if request.form.get('price') else None
            
            # Commit changes
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Flight updated successfully!'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Error updating flight: {str(e)}'}), 400
    
    return render_template('edit_flight.html', flight=flight.to_dict())


@app.route('/flights/delete/<int:id>', methods=['POST'])
@login_required
@cross_origin()
def delete_flight(id):
    """Delete a flight"""
    # 获取航班信息
    flight = Flight.query.get_or_404(id)
    
    # 检查权限：管理员可以删除任何航班，普通用户只能删除自己的航班
    user = User.query.get(session['user_id'])
    if not user.is_admin() and flight.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Access denied. You can only delete your own flights.'}), 403
    
    try:
        db.session.delete(flight)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Flight deleted successfully!'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error deleting flight: {str(e)}'}), 400


@app.route('/admin/users')
@admin_required
@cross_origin()
def admin_users():
    """Admin user management page"""
    try:
        # 只有管理员可以访问此页面
        current_user = User.query.get(session['user_id'])
        users = User.query.all()
        return render_template('admin_users.html', users=users, current_user=current_user)
    except Exception as e:
        return render_template('admin_users.html', error=str(e))


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

            # Check if arrival time is later than departure time
            dep_datetime = pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M")
            arr_datetime = pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M")
            
            if arr_datetime <= dep_datetime:
                return render_template('home.html', 
                                       prediction_text="Error: Arrival time must be later than departure time.",
                                       form_data=request.form)

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
            
            # Save the prediction to database
            try:
                # Create a new flight record
                dep_time = datetime.strptime(date_dep, "%Y-%m-%dT%H:%M")
                arr_time = datetime.strptime(date_arr, "%Y-%m-%dT%H:%M")
                
                # 如果用户已登录，则关联到用户
                user_id = session.get('user_id') if 'user_id' in session else None
                
                flight = Flight(
                    airline=airline,
                    source=Source,
                    destination=Destination,
                    departure_time=dep_time,
                    arrival_time=arr_time,
                    total_stops=Total_stops,
                    price=output,
                    user_id=user_id
                )
                
                db.session.add(flight)
                db.session.flush()  # Get the flight ID without committing
                
                # Create prediction record
                prediction_record = Prediction(
                    flight_id=flight.id,
                    predicted_price=output
                )
                
                db.session.add(prediction_record)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error: {db_error}")

            # Prepare flight details to pass to template
            flight_details = {
                'airline': airline,
                'source': Source,
                'destination': Destination,
                'departure_time': date_dep.replace('T', ' '),
                'arrival_time': date_arr.replace('T', ' '),
                'duration': f"{dur_hour}h {dur_min}m",
                'stops': Total_stops
            }

            # Prepare form data to refill the form
            form_data = {
                'Dep_Time': date_dep,
                'Arrival_Time': date_arr,
                'Source': Source,
                'Destination': Destination,
                'stops': Total_stops,
                'airline': airline
            }

            return render_template('home.html',
                                   prediction_text="Your Flight price is Rs. {}".format(output),
                                   prediction_value=output,
                                   price_stats=price_stats,
                                   flight_details=flight_details,
                                   form_data=form_data)
        except Exception as e:
            return render_template('home.html',prediction_text="Error occurred during prediction. Please check your inputs.")


    return render_template("home.html")




@app.route("/analysis")
@cross_origin()
def analysis():
    try:
        return render_template('analysis.html')
    except Exception as e:
        return render_template('analysis.html', error=str(e))

@app.route("/analysis/data")
@cross_origin()
def analysis_data():
    try:
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, '1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices', 'Data_Train.xlsx')
        df = pd.read_excel(data_path)
        df.columns = df.columns.str.strip()
        def pick(*names):
            for n in names:
                if n in df.columns:
                    return n
            return None
        col_airline = pick('Airline', 'airline')
        col_source = pick('Source', 'source')
        col_destination = pick('Destination', 'destination')
        col_stops = pick('Total_Stops', 'Total Stops', 'total_stops')
        col_price = pick('Price', 'price')
        col_date = pick('Date_of_Journey', 'Journey_date', 'Date')
        col_dep = pick('Dep_Time', 'Departure_Time', 'Dep_Time_full')
        if col_date is not None:
            df['_month'] = pd.to_datetime(df[col_date], errors='coerce', dayfirst=True).dt.month
        elif col_dep is not None:
            df['_month'] = pd.to_datetime(df[col_dep], errors='coerce', dayfirst=True).dt.month
        else:
            df['_month'] = np.nan
        if col_price is not None:
            df['_price'] = pd.to_numeric(df[col_price], errors='coerce')
        else:
            df['_price'] = np.nan
        airline_stats = []
        if col_airline is not None and df['_price'].notna().any():
            grp = df.dropna(subset=['_price']).groupby(col_airline)['_price'].mean().sort_values(ascending=False)
            airline_stats = [{'airline': k, 'avg_price': float(v)} for k, v in grp.items()]
        monthly_stats = []
        for m in range(1, 13):
            sel = df[(df['_month'] == m) & (df['_price'].notna())]
            avg = float(sel['_price'].mean()) if not sel.empty else 0.0
            monthly_stats.append({'month': m, 'avg_price': avg})
        pair_stats = []
        if col_source is not None and col_destination is not None and df['_price'].notna().any():
            df['_route'] = df[col_source].astype(str) + ' -> ' + df[col_destination].astype(str)
            grp = df.dropna(subset=['_price']).groupby('_route')['_price'].mean().sort_values(ascending=False).head(15)
            pair_stats = [{'route': k, 'avg_price': float(v)} for k, v in grp.items()]
        stops_stats = []
        if col_stops is not None and df['_price'].notna().any():
            grp = df.dropna(subset=['_price']).groupby(col_stops)['_price'].mean().sort_values()
            stops_stats = [{'stops': int(k), 'avg_price': float(v)} for k, v in grp.items()]
        price_series = df['_price'].dropna()
        if not price_series.empty:
            hist_counts, bin_edges = np.histogram(price_series, bins=20)
            price_dist = {'bins': [float(x) for x in bin_edges], 'counts': [int(x) for x in hist_counts]}
        else:
            price_dist = {'bins': [], 'counts': []}
        return jsonify({
            'airline_prices': airline_stats,
            'monthly_prices': monthly_stats,
            'source_dest_prices': pair_stats,
            'stops_prices': stops_stats,
            'price_distribution': price_dist
        })
    except Exception as e:
        try:
            flights = Flight.query.all()
            prices = [float(f.price) for f in flights if f.price is not None]
            airline_map = {}
            for f in flights:
                if f.price is None:
                    continue
                k = f.airline
                airline_map.setdefault(k, []).append(float(f.price))
            airline_stats = [{'airline': k, 'avg_price': float(np.mean(v))} for k, v in airline_map.items()]
            airline_stats.sort(key=lambda x: x['avg_price'], reverse=True)
            month_map = {}
            for f in flights:
                if f.price is None or f.departure_time is None:
                    continue
                m = f.departure_time.month
                month_map.setdefault(m, []).append(float(f.price))
            monthly_stats = []
            for m in range(1, 13):
                if m in month_map:
                    monthly_stats.append({'month': m, 'avg_price': float(np.mean(month_map[m]))})
                else:
                    monthly_stats.append({'month': m, 'avg_price': 0.0})
            pair_map = {}
            for f in flights:
                if f.price is None:
                    continue
                label = f"{f.source} -> {f.destination}"
                pair_map.setdefault(label, []).append(float(f.price))
            pair_stats = [{'route': k, 'avg_price': float(np.mean(v))} for k, v in pair_map.items()]
            pair_stats.sort(key=lambda x: x['avg_price'], reverse=True)
            pair_stats = pair_stats[:15]
            stops_map = {}
            for f in flights:
                if f.price is None:
                    continue
                s = int(f.total_stops)
                stops_map.setdefault(s, []).append(float(f.price))
            stops_stats = [{'stops': s, 'avg_price': float(np.mean(v))} for s, v in sorted(stops_map.items())]
            if prices:
                bins = 20
                hist_counts, bin_edges = np.histogram(prices, bins=bins)
                price_dist = {
                    'bins': [float(x) for x in bin_edges],
                    'counts': [int(x) for x in hist_counts]
                }
            else:
                price_dist = {'bins': [], 'counts': []}
            return jsonify({
                'airline_prices': airline_stats,
                'monthly_prices': monthly_stats,
                'source_dest_prices': pair_stats,
                'stops_prices': stops_stats,
                'price_distribution': price_dist,
                'warning': str(e)
            }), 200
        except Exception as inner:
            return jsonify({'error': str(e), 'fallback_error': str(inner)}), 500


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
