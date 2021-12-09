from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    section_id : str
    patch_id : str
    timestamp : Optional[str] = None