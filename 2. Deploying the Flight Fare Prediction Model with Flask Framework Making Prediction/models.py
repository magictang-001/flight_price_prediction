from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Flight(db.Model):
    __tablename__ = 'flights'
    
    id = db.Column(db.Integer, primary_key=True)
    airline = db.Column(db.String(100), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    destination = db.Column(db.String(100), nullable=False)
    departure_time = db.Column(db.DateTime, nullable=False)
    arrival_time = db.Column(db.DateTime, nullable=False)
    total_stops = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Numeric(precision=10, scale=2))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='flight', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Flight {self.airline} from {self.source} to {self.destination}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'airline': self.airline,
            'source': self.source,
            'destination': self.destination,
            'departure_time': self.departure_time.isoformat() if self.departure_time else None,
            'arrival_time': self.arrival_time.isoformat() if self.arrival_time else None,
            'total_stops': self.total_stops,
            'price': float(self.price) if self.price else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    flight_id = db.Column(db.Integer, db.ForeignKey('flights.id', ondelete='SET NULL'), nullable=True)
    predicted_price = db.Column(db.Numeric(precision=10, scale=2), nullable=False)
    actual_price = db.Column(db.Numeric(precision=10, scale=2), nullable=True)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction for Flight {self.flight_id}: {self.predicted_price}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'flight_id': self.flight_id,
            'predicted_price': float(self.predicted_price),
            'actual_price': float(self.actual_price) if self.actual_price else None,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None
        }

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }