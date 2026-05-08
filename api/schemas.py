from pydantic import BaseModel
from typing import Optional, List


class ForecastRequest(BaseModel):
    sku_id: str
    distribution_center: str
    horizon: int = 7


class ForecastResponse(BaseModel):
    sku_id: str
    distribution_center: str
    horizon: int
    forecasts: List[float]
    lower_bound: List[float]
    upper_bound: List[float]


class MetricsResponse(BaseModel):
    model: str
    WMAPE: float
    MAE: float
    RMSE: float


class FeatureImportanceResponse(BaseModel):
    top_features: List[dict]


class RetrainResponse(BaseModel):
    status: str
    message: str