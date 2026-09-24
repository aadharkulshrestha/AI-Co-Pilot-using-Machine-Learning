"""
SQLAlchemy Database Models for AI Co-Pilot Aviation Platform
"""

import time
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Flight(Base):
    __tablename__ = "flights"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(50), unique=True, index=True)
    callsign = Column(String(50))
    aircraft_type = Column(String(100))
    origin = Column(String(50))
    destination = Column(String(50))
    status = Column(String(50), default="IN_FLIGHT")
    created_at = Column(Float, default=time.time)
    
    telemetry_records = relationship("TelemetryRecord", back_populates="flight", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="flight", cascade="all, delete-orphan")


class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(50), ForeignKey("flights.flight_id"), index=True)
    timestamp = Column(Float, default=time.time)
    altitude = Column(Float)
    airspeed = Column(Float)
    vertical_rate = Column(Float)
    pitch = Column(Float)
    roll = Column(Float)
    heading = Column(Float)
    throttle = Column(Float)
    g_force = Column(Float)
    lat = Column(Float)
    lon = Column(Float)
    
    flight = relationship("Flight", back_populates="telemetry_records")


class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(50), unique=True, index=True)
    flight_id = Column(String(50), index=True)
    aircraft_type = Column(String(100))
    origin = Column(String(50))
    destination = Column(String(50))
    event_type = Column(String(100))
    flight_phase = Column(String(50))
    severity = Column(String(50))
    narrative = Column(Text)
    outcome = Column(String(100))
    risk_score = Column(Float)
    predicted_action = Column(String(100))
    created_at = Column(Float, default=time.time)


class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(50), ForeignKey("flights.flight_id"), index=True)
    timestamp = Column(Float, default=time.time)
    abnormal_event = Column(String(100))
    predicted_action = Column(String(100))
    confidence = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String(50))
    recommendation = Column(Text)
    
    flight = relationship("Flight", back_populates="predictions")
