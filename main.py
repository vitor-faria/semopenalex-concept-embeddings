from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Concept(BaseModel):
    prefLabel: str
    year: int
    neighbors: Union[int, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/concept/{concept_id}")
def read_concept(concept_id: int, q: Union[str, None] = None):
    return {"concept_id": concept_id, "q": q}

