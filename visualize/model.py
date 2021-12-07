from typing import List
from pydantic import BaseModel


class Patch(BaseModel):
    patch_id: int
    patch_RLE: str


class Section(BaseModel):
    section_id: int
    section_flag: bool
    patches: List[Patch]


class Model(BaseModel):
    timestamp: str
    sections: List[Section]
