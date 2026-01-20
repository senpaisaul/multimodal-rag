from pydantic import BaseModel
from typing import List, Literal, Optional


class VisionFact(BaseModel):
    page: int
    image_type: Literal["chart", "table", "diagram", "text", "other"]
    description: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    data_points: List[List[float]] = []
    trend: str = ""
    confidence: Literal["high", "medium", "low"]
