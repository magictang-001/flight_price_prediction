-- MySQL Database Setup Script for Flight Price Prediction Application

-- Create database
CREATE DATABASE IF NOT EXISTS flight_prediction_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Use the database
USE flight_prediction_db;

-- Create users table for application users (needs to be created before flights table due to foreign key constraint)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create flights table to store flight information
CREATE TABLE IF NOT EXISTS flights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    airline VARCHAR(100) NOT NULL,
    source VARCHAR(100) NOT NULL,
    destination VARCHAR(100) NOT NULL,
    departure_time DATETIME NOT NULL,
    arrival_time DATETIME NOT NULL,
    total_stops INT NULL,
    aircraft_model VARCHAR(200) NULL,
    price DECIMAL(10, 2),
    user_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
--     FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Create predictions table to store prediction records
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    flight_id INT,
    predicted_price DECIMAL(10, 2) NOT NULL,
    actual_price DECIMAL(10, 2) NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
--     FOREIGN KEY (flight_id) REFERENCES flights(id) ON DELETE SET NULL
);

-- Insert sample data for testing
-- Password is 'admin123' for both users
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PZvO.S', 'admin'),
('user1', 'user1@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PZvO.S', 'user');

-- INSERT INTO flights (airline, source, destination, departure_time, arrival_time, total_stops, price, user_id) VALUES
-- ('IndiGo', 'Delhi', 'Bangalore', '2025-12-15 08:00:00', '2025-12-15 10:30:00', 0, 4500.00, 2),
-- ('Air India', 'Mumbai', 'Chennai', '2025-12-16 14:00:00', '2025-12-16 16:45:00', 1, 7200.50, 2),
-- ('Jet Airways', 'Kolkata', 'Delhi', '2025-12-17 19:30:00', '2025-12-17 22:15:00', 0, 8900.75, NULL);
-- Migration helper for existing databases:
-- ALTER TABLE flights MODIFY total_stops INT NULL;
-- ALTER TABLE flights ADD COLUMN aircraft_model VARCHAR(200) NULL;
