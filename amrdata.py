from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class AmrData(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float