import json
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from src.functions import get_results

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
    max_depth: Union[int, None] = None,
    max_walks: Union[int, None] = None,
    with_reverse: Union[bool, None] = None,
    random_seed: Union[int, None] = None,
):
    response = {"concept_id": concept_id}

    year = year if year else 2022
    response.update({"year": year})

    neighbors = neighbors if neighbors else 10
    response.update({"neighbors": neighbors})

    limit = limit if limit else 1000
    response.update({"limit": limit})

    max_depth = max_depth if max_depth else 6
    response.update({"max_depth": max_depth})

    max_walks = max_walks if max_walks else 12
    response.update({"max_walks": max_walks})

    with_reverse = with_reverse if with_reverse else True
    response.update({"with_reverse": with_reverse})

    random_seed = random_seed if random_seed else 42
    response.update({"random_seed": random_seed})

    if all([
        concept_id == "C121608353",
        year == 2023,
        neighbors == 5,
    ]):
        with open('src/toy_example_output.json', 'r') as file:
            toy_example = json.load(file)
            response = toy_example

    else:
        response = get_results(
            broader_concept_id=concept_id,
            year=year,
            limit=limit,
            max_depth=max_depth,
            max_walks=max_walks,
            with_reverse=with_reverse,
            random_seed=random_seed,
            k=neighbors,
        )

    return response
