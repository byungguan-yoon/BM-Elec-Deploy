from pydantic import BaseModel

class Item(BaseModel):
    section_id : str
    patch_id : str