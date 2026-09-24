import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Co-Pilot: Aviation Flight Risk & Predictive Decision-Making"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    
    # Storage and Models
    SAVED_MODELS_DIR: Path = BASE_DIR / "saved_models"
    SAMPLE_DATA_DIR: Path = BASE_DIR.parent / "sample_data"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/copilot.db")
    
    # ML & Simulation Settings
    SEQUENCE_LENGTH: int = 15  # 15 time-steps for sequential model
    FEATURE_COUNT: int = 12
    SIMULATION_TICK_RATE_HZ: float = 2.0  # Live stream tick frequency
    
    # Risk thresholds
    RISK_LOW_THRESHOLD: float = 25.0
    RISK_MEDIUM_THRESHOLD: float = 60.0
    RISK_HIGH_THRESHOLD: float = 80.0
    
    class Config:
        case_sensitive = True

settings = Settings()
settings.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
