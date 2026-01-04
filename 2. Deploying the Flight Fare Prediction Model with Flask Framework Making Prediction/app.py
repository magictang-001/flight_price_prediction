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
from models import db, Flight, Prediction, User, TicketData
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

# Currency conversion
app.config['INR_TO_CNY'] = float(os.environ.get('INR_TO_CNY', '0.086'))
# Create tables
with app.app_context():
    db.create_all()

# CSV import helper
def import_csv_to_db(limit=None):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, '1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices')
        csv_path = os.path.join(data_dir, '机票数据.csv')
        df = None
        if os.path.exists(csv_path):
            for enc in ['utf-8-sig', 'utf-8', 'gbk']:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    break
                except Exception:
                    continue
            if df is None:
                df = pd.read_csv(csv_path)
        else:
            return {'success': False, 'message': 'CSV file not found'}
        df.columns = df.columns.str.strip()
        def pick(*names):
            for n in names:
                if n in df.columns:
                    return n
            return None
        c_source = pick('出发城市','Source')
        c_dest = pick('到达城市','Destination')
        c_date = pick('出发日期','Date_of_Journey')
        c_airline = pick('航空公司','Airline')
        c_model = pick('客机机型','Aircraft_Model')
        c_dep_air = pick('出发机场')
        c_arr_air = pick('到达机场')
        c_dep_time = pick('出发时间','Dep_Time')
        c_arr_time = pick('抵达时间','Arrival_Time')
        c_cabin = pick('客舱类型')
        c_price = pick('机票价格','Price')
        rows = df if limit is None else df.head(limit)
        objs = []
        for _, r in rows.iterrows():
            raw_price = str(r[c_price]) if c_price else ''
            cleaned = ''.join([ch for ch in raw_price if (ch.isdigit() or ch=='.')])
            price_num = None
            try:
                price_num = float(cleaned) if cleaned!='' else None
            except Exception:
                price_num = None
            obj = TicketData(
                source_city=str(r[c_source]) if c_source else None,
                destination_city=str(r[c_dest]) if c_dest else None,
                departure_date=str(r[c_date]) if c_date else None,
                airline=str(r[c_airline]) if c_airline else None,
                aircraft_model=str(r[c_model]) if c_model else None,
                departure_airport=str(r[c_dep_air]) if c_dep_air else None,
                arrival_airport=str(r[c_arr_air]) if c_arr_air else None,
                departure_time=str(r[c_dep_time]) if c_dep_time else None,
                arrival_time=str(r[c_arr_time]) if c_arr_time else None,
                cabin_type=str(r[c_cabin]) if c_cabin else None,
                price=price_num,
                raw_price_text=raw_price
            )
            objs.append(obj)
        if len(objs) == 0:
            return {'success': False, 'message': 'No rows to import'}
        db.session.bulk_save_objects(objs)
        db.session.commit()
        return {'success': True, 'imported': len(objs)}
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'message': str(e)}
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
    data_dir = os.path.join(project_root, '1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices')
    csv_path = os.path.join(data_dir, '机票数据.csv')
    xlsx_path = os.path.join(data_dir, 'Data_Train.xlsx')
    df_train = None
    if os.path.exists(csv_path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk']:
            try:
                df_train = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception:
                continue
        if df_train is None:
            df_train = pd.read_csv(csv_path)
        df_train.columns = df_train.columns.str.strip()
        price_col = 'Price' if 'Price' in df_train.columns else ('机票价格' if '机票价格' in df_train.columns else None)
        if price_col:
            cleaned = df_train[price_col].astype(str).str.replace(r'[^0-9\.]', '', regex=True)
            ps = pd.to_numeric(cleaned, errors='coerce').dropna()
            if not ps.empty:
                price_stats = {
                    'min': float(ps.min()),
                    'max': float(ps.max()),
                    'mean': float(ps.mean()),
                    '25%': float(ps.quantile(0.25)),
                    '50%': float(ps.quantile(0.50)),
                    '75%': float(ps.quantile(0.75))
                }
                print("Price statistics loaded from 机票数据.csv")
    elif os.path.exists(xlsx_path):
        try:
            df_train = pd.read_excel(xlsx_path)
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

# Prepare dropdown options from CSV if available
dropdown_options = {
    'sources': ['Delhi', 'Kolkata', 'Mumbai', 'Chennai'],
    'destinations': ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'],
    'airlines': ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet',
                 'Vistara', 'Air Asia', 'GoAir', 'Multiple carriers Premium economy',
                 'Jet Airways Business', 'Vistara Premium economy', 'Trujet'],
    'aircraft_models': []
}
try:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices')
    csv_path = os.path.join(data_dir, '机票数据.csv')
    if os.path.exists(csv_path):
        df_opts = None
        for enc in ['utf-8-sig', 'utf-8', 'gbk']:
            try:
                df_opts = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception:
                continue
        if df_opts is None:
            df_opts = pd.read_csv(csv_path)
        df_opts.columns = df_opts.columns.str.strip()
        def uniques(col_names):
            for n in col_names:
                if n in df_opts.columns:
                    vals = df_opts[n].dropna().astype(str).str.strip()
                    return sorted(list(dict.fromkeys(vals.tolist())))
            return None
        sources = uniques(['出发城市', 'Source'])
        destinations = uniques(['到达城市', 'Destination'])
        airlines = uniques(['航空公司', 'Airline'])
        aircraft_models = uniques(['客机机型', 'Aircraft_Model'])
        if sources and len(sources) > 0:
            dropdown_options['sources'] = sources
        if destinations and len(destinations) > 0:
            dropdown_options['destinations'] = destinations
        if airlines and len(airlines) > 0:
            dropdown_options['airlines'] = airlines
        if aircraft_models and len(aircraft_models) > 0:
            dropdown_options['aircraft_models'] = aircraft_models
        print("Dropdown options loaded from 机票数据.csv")
except Exception as e:
    print(f"Could not load dropdown options from CSV: {e}")


@app.route("/")
@cross_origin()
def home():
    if model is None:
        return render_template("home.html", 
                               prediction_text="Model not loaded. The application is running, but predictions cannot be made. This is likely due to a version incompatibility with the trained model. Please contact the administrator to retrain the model.",
                               dropdown_options=dropdown_options)
    return render_template("home.html", dropdown_options=dropdown_options)

@app.route('/admin/import_csv')
@admin_required
@cross_origin()
def import_csv_route():
    res = import_csv_to_db()
    if res.get('success'):
        return jsonify(res)
    return jsonify(res), 400

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
    return render_template("home.html", prediction_text="Page not found.", dropdown_options=dropdown_options), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("home.html", prediction_text="Internal server error. Please try again.", dropdown_options=dropdown_options), 500


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
        distinct_aircraft_models = [r.aircraft_model for r in query.with_entities(Flight.aircraft_model).distinct().order_by(Flight.aircraft_model).all()]
            
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
            
        aircraft_model = request.args.get('aircraft_model')
        if aircraft_model:
            query = query.filter(Flight.aircraft_model.ilike(f"%{aircraft_model}%"))

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
                             destinations=distinct_destinations,
                             aircraft_models=distinct_aircraft_models)
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
            aircraft_model = request.form.get('aircraft_model')
            price = float(request.form.get('price')) if request.form.get('price') else None
            
            # Create new flight
            flight = Flight(
                airline=airline,
                source=source,
                destination=destination,
                departure_time=departure_time,
                arrival_time=arrival_time,
                total_stops=None,
                aircraft_model=aircraft_model,
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
    
    return render_template('add_flight.html',
                           aircraft_models=dropdown_options.get('aircraft_models', []),
                           airlines=dropdown_options.get('airlines', []),
                           sources=dropdown_options.get('sources', []),
                           destinations=dropdown_options.get('destinations', []))


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
            flight.aircraft_model = request.form.get('aircraft_model')
            flight.price = float(request.form.get('price')) if request.form.get('price') else None
            
            # Commit changes
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Flight updated successfully!'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Error updating flight: {str(e)}'}), 400
    
    return render_template('edit_flight.html',
                           flight=flight.to_dict(),
                           aircraft_models=dropdown_options.get('aircraft_models', []),
                           airlines=dropdown_options.get('airlines', []),
                           sources=dropdown_options.get('sources', []),
                           destinations=dropdown_options.get('destinations', []))


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
            dep_str = request.form["Dep_Time"]
            arr_str = request.form["Arrival_Time"]
            dep_dt = pd.to_datetime(dep_str, format="%Y-%m-%dT%H:%M")
            arr_dt = pd.to_datetime(arr_str, format="%Y-%m-%dT%H:%M")
            Journey_day = int(dep_dt.day)
            Journey_month = int(dep_dt.month)
            Dep_hour = int(dep_dt.hour)
            Dep_min = int(dep_dt.minute)
            Arrival_hour = int(arr_dt.hour)
            Arrival_min = int(arr_dt.minute)
            dep_total_min = Dep_hour * 60 + Dep_min
            arr_total_min = Arrival_hour * 60 + Arrival_min
            delta_min = arr_total_min - dep_total_min
            if delta_min < 0:
                delta_min += 24 * 60
            dur_hour = int(delta_min // 60)
            dur_min = int(delta_min % 60)
        except Exception:
            return render_template('home.html', prediction_text="Invalid date/time format. Please check your inputs.", dropdown_options=dropdown_options)

        airline = request.form.get('airline', '')
        Source = request.form.get('Source', '')
        Destination = request.form.get('Destination', '')
        aircraft_model = request.form.get('aircraft_model', '')
        Total_stops = 0
        if not airline or not Source or not Destination or not aircraft_model:
            form_data = {
                'Dep_Time': dep_str,
                'Arrival_Time': arr_str,
                'Source': Source,
                'Destination': Destination,
                'airline': airline,
                'aircraft_model': aircraft_model
            }
            return render_template('home.html',
                                   prediction_text="请输入完整的出发/到达时间、城市、航空公司与客机类型。",
                                   form_data=form_data,
                                   dropdown_options=dropdown_options)

        try:
            df = pd.DataFrame([{
                "Journey_day": Journey_day,
                "Journey_month": Journey_month,
                "Dep_hour": Dep_hour,
                "Dep_min": Dep_min,
                "Arrival_hour": Arrival_hour,
                "Arrival_min": Arrival_min,
                "Duration_hours": dur_hour,
                "Duration_mins": dur_min,
                "Airline": airline,
                "Source": Source,
                "Destination": Destination,
                "Aircraft_Model": aircraft_model
            }])
            pred_inr = float(model.predict(df)[0])
            price_cny = round(pred_inr * app.config.get('INR_TO_CNY', 0.086), 2)

            try:
                dep_time = datetime.strptime(dep_str, "%Y-%m-%dT%H:%M")
                arr_time = datetime.strptime(arr_str, "%Y-%m-%dT%H:%M")
                user_id = session.get('user_id') if 'user_id' in session else None
                flight = Flight(
                    airline=airline,
                    source=Source,
                    destination=Destination,
                    departure_time=dep_time,
                    arrival_time=arr_time,
                    total_stops=Total_stops,
                    aircraft_model=aircraft_model,
                    price=price_cny,
                    user_id=user_id
                )
                db.session.add(flight)
                db.session.flush()
                prediction_record = Prediction(
                    flight_id=flight.id,
                    predicted_price=price_cny
                )
                db.session.add(prediction_record)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error: {db_error}")

            flight_details = {
                'airline': airline,
                'source': Source,
                'destination': Destination,
                'departure_time': dep_str.replace('T', ' '),
                'arrival_time': arr_str.replace('T', ' '),
                'duration': f"{dur_hour}h {dur_min}m",
                'aircraft_model': aircraft_model
            }
            form_data = {
                'Dep_Time': dep_str,
                'Arrival_Time': arr_str,
                'Source': Source,
                'Destination': Destination,
                'airline': airline,
                'aircraft_model': aircraft_model
            }
            stats_cny = {k: round(v * app.config.get('INR_TO_CNY', 0.086), 2) for k, v in price_stats.items()}
            return render_template('home.html',
                                   prediction_text="您的航班价格为 ￥ {}".format(price_cny),
                                   prediction_value=price_cny,
                                   price_stats=stats_cny,
                                   flight_details=flight_details,
                                   form_data=form_data,
                                   dropdown_options=dropdown_options)
        except Exception as e:
            msg = str(e)
            if "ColumnTransformer" in msg and "_name_to_fitted_passthrough" in msg:
                try:
                    model_reload = pickle.load(open("flight_xgb.pkl", "rb"))
                    pred_inr = float(model_reload.predict(df)[0])
                    price_cny = round(pred_inr * app.config.get('INR_TO_CNY', 0.086), 2)
                    flight_details = {
                        'airline': airline,
                        'source': Source,
                        'destination': Destination,
                        'departure_time': dep_str.replace('T', ' '),
                        'arrival_time': arr_str.replace('T', ' '),
                        'duration': f"{dur_hour}h {dur_min}m",
                        'aircraft_model': aircraft_model
                    }
                    form_data = {
                        'Dep_Time': dep_str,
                        'Arrival_Time': arr_str,
                        'Source': Source,
                        'Destination': Destination,
                        'airline': airline,
                        'aircraft_model': aircraft_model
                    }
                    stats_cny = {k: round(v * app.config.get('INR_TO_CNY', 0.086), 2) for k, v in price_stats.items()}
                    return render_template('home.html',
                                           prediction_text="您的航班价格为 ￥ {}".format(price_cny),
                                           prediction_value=price_cny,
                                           price_stats=stats_cny,
                                           flight_details=flight_details,
                                           form_data=form_data,
                                           dropdown_options=dropdown_options)
                except Exception as e2:
                    msg = str(e2)
            form_data = {
                'Dep_Time': dep_str,
                'Arrival_Time': arr_str,
                'Source': Source,
                'Destination': Destination,
                'airline': airline,
                'aircraft_model': aircraft_model
            }
            return render_template('home.html',
                                   prediction_text=f"预测失败，请检查输入。详情：{msg}",
                                   form_data=form_data,
                                   dropdown_options=dropdown_options)


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
        data_dir = os.path.join(base_dir, '1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices')
        csv_path = os.path.join(data_dir, '机票数据.csv')
        xlsx_path = os.path.join(data_dir, 'Data_Train.xlsx')
        if os.path.exists(csv_path):
            for enc in ['utf-8-sig', 'utf-8', 'gbk']:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    break
                except Exception:
                    continue
            else:
                df = pd.read_csv(csv_path)
        elif os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
        else:
            df = pd.DataFrame()
        df.columns = df.columns.str.strip()
        def pick(*names):
            for n in names:
                if n in df.columns:
                    return n
            return None
        col_airline = pick('Airline', 'airline', '航空公司')
        col_source = pick('Source', 'source', '出发城市')
        col_destination = pick('Destination', 'destination', '到达城市')
        col_stops = pick('Total_Stops', 'Total Stops', 'total_stops', '经停次数')
        col_price = pick('Price', 'price', '机票价格')
        col_date = pick('Date_of_Journey', 'Journey_date', 'Date', '出发日期')
        col_dep = pick('Dep_Time', 'Departure_Time', 'Dep_Time_full', '出发时间')
        def extract_month(val):
            import re
            s = str(val).strip()
            if s == '':
                return np.nan
            m = re.search(r'(\d{1,2})月', s)
            if m:
                mm = int(m.group(1))
                if 1 <= mm <= 12:
                    return mm
            m2 = re.search(r'(\d{1,2})[\\-/](\d{1,2})', s)
            if m2:
                mm = int(m2.group(1))
                dd = int(m2.group(2))
                if 1 <= mm <= 12 and 1 <= dd <= 31:
                    return mm
            try:
                dt = pd.to_datetime(s, errors='coerce', dayfirst=True)
                if pd.notna(dt):
                    return int(dt.month)
            except Exception:
                pass
            return np.nan
        if col_date is not None:
            df['_month'] = df[col_date].apply(extract_month)
        elif col_dep is not None:
            def dep_has_date(strval):
                import re
                t = str(strval).strip()
                if t == '':
                    return False
                if re.search(r'\\d{4}[\\-/]\\d{1,2}[\\-/]\\d{1,2}', t):
                    return True
                if ('年' in t) and ('月' in t):
                    return True
                if '月' in t:
                    return True
                if re.search(r'\\d{1,2}[\\-/]\\d{1,2}', t):
                    return True
                return False
            mask = df[col_dep].apply(dep_has_date)
            df['_month'] = df[col_dep].where(mask).apply(extract_month)
        else:
            df['_month'] = np.nan
        if col_price is not None:
            cleaned = df[col_price].astype(str).str.replace(r'[^0-9\.]', '', regex=True)
            df['_price'] = pd.to_numeric(cleaned, errors='coerce')
        else:
            df['_price'] = np.nan
        col_stops_raw = col_stops
        if col_stops_raw is not None:
            def parse_stops(v):
                s = str(v).strip()
                if s == '':
                    return np.nan
                try:
                    return int(s)
                except Exception:
                    import re
                    m = re.search(r'(\d+)', s)
                    if m:
                        return int(m.group(1))
                    if s in ['直飞', 'Non-Stop', 'none', '无经停']:
                        return 0
                    return np.nan
            df['_stops_parsed'] = df[col_stops_raw].apply(parse_stops)
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
            key = '_stops_parsed' if '_stops_parsed' in df.columns else col_stops
            grp = df.dropna(subset=['_price']).dropna(subset=[key]).groupby(key)['_price'].mean().sort_values()
            try:
                stops_stats = [{'stops': int(k), 'avg_price': float(v)} for k, v in grp.items()]
            except Exception:
                stops_stats = [{'stops': str(k), 'avg_price': float(v)} for k, v in grp.items()]
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
