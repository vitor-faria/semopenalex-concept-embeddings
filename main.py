import json
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
def read_concept(
    concept_id: str,
    year: int,
    neighbors: Union[int, None] = None,
    limit: Union[int, None] = None,
):
    response = {"concept_id": concept_id}

    year = year if year else 2022
    response.update({"year": year})

    neighbors = neighbors if neighbors else 10
    response.update({"neighbors": neighbors})

    limit = limit if limit else 1000
    response.update({"limit": limit})

    if all([
        concept_id == "C121608353",
        year == 2023,
        neighbors == 5,
    ]):
        with open('src/toy_example_output.json', 'r') as file:
            toy_example = json.load(file)
            response = toy_example

    return response
