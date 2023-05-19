# SemOpenAlex Concept Embeddings

FastAPI application that calculates Knowledge Graph embeddings for a SemOpenAlex Concept 
using RDF2Vec, and returns similar concepts with their coordinates in 2-dimensional space.

## Development Environment

To create the development environment it's recommended to use conda.

Run the following commands to get the environment ready

```
conda create -n ENVIRONMENT_NAME python=3.9
conda activate ENVIRONMENT_NAME
pip install -r requirements.txt
```

### Running Locally

To run the FastAPI application in your local host use

```
uvicorn app.main:app --reload
```

### Running on Docker

To build the Docker image, run 

```
docker build -t conceptembs .
```

Then run the Docker container with 

```
docker run -d --name conceptembscont -p 80:80 conceptembs
```

## Getting embeddings

The endpoint used to calculate KG embeddings for concepts related to a seed concept, 
and get the closest concepts to this seed is 

```
/embeddings?concept_id=[CONCEPT_ID]
```

Note that `concept_id` is a mandatory query parameter. Other optional parameters are:

- `year`: Year used to filter works in the SPARQL query against SemOpenAlex' endpoint.
Defaults to `2022`.
- `limit`: Number of triples fetched in the SPARQL query against SemOpenAlex' endpoint.
The more triples, the better the embeddings are likely to be, but the longer it might take 
to generate RDF2Vec walks. Defaults to `1000`.
- `neighbors`: Number of nearest neighbors in the result set. The seed concept `concept_id` 
is included in this set, and is always the first element. Defaults to `10`.
- `max_depth`: Maximum depth of the walks generated in RDF2Vec. The deeper the walks, 
the better the embeddings are likely to be, but the longer it might take 
to generate walks. Defaults to `6`.
- `max_walks`: Maximum number of the walks per starting entity generated in RDF2Vec. 
The more walks, the better the embeddings are likely to be, but the longer it might take 
to generate walks. Defaults to `12`.
- `with_reverse`: Boolean that allows or not reverse walking in RDF2Vec. 
Defaults to `true`.
- `random_seed`: Random seed passed to both `RandomWalker` and `Word2Vec` steps of RDF2Vec.
Defaults to `42`.

### Toy example output

One specific pre-computed example is returned for the following query:

```
/embeddings?concept_id=C154945302&limit=20000
```

This toy example (Concept [Artificial Intelligence](https://semopenalex.org/concept/C154945302), 
with 20k triples and default parameters) is materialized in a file, so the API will not execute any pipeline 
step. This is to illustrate how results are presented, and also to easily check if the 
API is available without computing embeddings.

## Production endpoint

A Flask version of this project is available in the following endpoint:

`https://concept-embeddings-fo6ivoj5aa-uc.a.run.app/`

The toy example output can than be fetched by the following GET request:

https://concept-embeddings-fo6ivoj5aa-uc.a.run.app/?concept_id=C154945302&limit=20000

### Metaphactory integration

The production endpoint can be integrated to a Metaphactory instance by adding the following configurations:

- Add a repository `conceptembs` with 
[this content](metaphactory_templates/conceptembs_repository.ttl)
- Update the repository `defaultEphedra` with 
[this content](metaphactory_templates/default_ephedra_repository.ttl)
- Add a Ephedra service `conceptembs` with 
[this content](metaphactory_templates/conceptembs_ephedra_service.ttl)

Done that, the API can be queried directly from Metaphactory with SPARQL. 
A SPARQL query that returns the toy example is in 
[this file](metaphactory_templates/toy_example_sparql.sparql).
