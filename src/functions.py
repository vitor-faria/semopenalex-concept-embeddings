import numpy as np
import pandas as pd
from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.samplers import PageRankSampler
from pyrdf2vec.walkers import RandomWalker
import rdflib
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from SPARQLWrapper import SPARQLWrapper, TURTLE, JSON


def get_kg(concept_id, year, limit=500000):
    url = "https://semopenalex.org/sparql"
    sparql = SPARQLWrapper(url)
    query_prefix = """
    PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
    PREFIX foaf:<http://xmlns.com/foaf/0.1/>
    PREFIX soap:<https://semopenalex.org/property/>
    """
    where_start = """
    CONSTRUCT
    WHERE {
        
    """
    where_patterns = f"""
        ?work1 soap:hasConcept <https://semopenalex.org/concept/{concept_id}> .  
        ?work1 <http://purl.org/spar/fabio/hasPublicationYear> {year} .
        ?work1 <http://purl.org/spar/cito/cites> ?work2 .
        ?work2 soap:hasConcept <https://semopenalex.org/concept/{concept_id}> .
        ?work1 soap:hasConcept ?topic1 .
        ?work1 <http://purl.org/dc/terms/creator> ?author1 .
        ?author1 <http://www.w3.org/ns/org#memberOf> ?institution1 .
        ?work2 soap:hasConcept ?topic2 .
    """
    where_end = """
    }
    LIMIT 
    """
    construct_query = query_prefix + where_start + where_patterns + where_end + str(limit)

    print(construct_query)

    sparql.setQuery(construct_query)
    sparql.setReturnFormat(TURTLE)
    results = sparql.query().convert()
    g = rdflib.Graph()
    g.parse(data=results, format="turtle")

    work_has_concept = rdflib.term.URIRef('https://semopenalex.org/property/hasConcept')
    concept_has_work = rdflib.term.URIRef('https://semopenalex.org/property/conceptHasWork')
    for work, relation, concept in g.triples((None, work_has_concept, None)):
        g.add((concept, concept_has_work, work))

    return g


def get_concepts_list(g):
    work_has_concept = rdflib.term.URIRef('https://semopenalex.org/property/hasConcept')
    concepts = set()
    for work, relation, concept in g.triples((None, work_has_concept, None)):
        concepts.add(str(concept))

    return [concept for concept in concepts]


def get_embeddings(g, max_depth=6, max_walks=12, with_reverse=True, random_seed=42):
    concepts = get_concepts_list(g)
    print(f'fetched {len(concepts)} concepts')

    if len(concepts) > 0:
        ttl_path = "src/temp/triples.ttl"
        with open(ttl_path, "w", encoding="utf-8") as file:
            file.write(g.serialize(format='turtle'))
        print('serialized turtle file')

        knowledge_graph = KG(
            ttl_path,
            skip_predicates={
                'http://purl.org/spar/fabio/hasPublicationYear',
            },
        )
        print('declared KG')

        transformer = RDF2VecTransformer(
            Word2Vec(
                seed=random_seed,
                workers=1,
            ),
            walkers=[RandomWalker(
                max_depth=max_depth,
                max_walks=max_walks,
                sampler=PageRankSampler(),
                with_reverse=with_reverse,
                md5_bytes=None,
                random_state=random_seed,
            )],
            verbose=2
        )
        print('declared Transformer')

        walks = transformer.get_walks(
            knowledge_graph,
            concepts,
        )
        print('finished walking')

        transformer.fit(walks)
        print('fitted walks')

        concepts_embeddings, literals = transformer.transform(
            kg=knowledge_graph,
            entities=concepts,
        )
        print('calculated embeddings')

        return concepts, concepts_embeddings

    else:
        return [], []


def get_k_nearest_neighbors(input_concept, concepts_list, concepts_embeddings, k=10):
    cosine_distance = cdist(concepts_embeddings, concepts_embeddings, 'cosine')
    input_index = concepts_list.index(input_concept)
    distance_array = cosine_distance[input_index]
    rank = np.argsort(np.argsort(distance_array))

    result_df = pd.DataFrame({
        "concept": concepts_list,
        "distance": distance_array,
        "rank": rank
    })

    return result_df.set_index('concept').sort_values(by="rank").head(k)


def get_concept_labels(year):  # TODO fix works count for current year
    url = "https://semopenalex.org/sparql"
    sparql = SPARQLWrapper(url)

    query_start = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX soap:<https://semopenalex.org/property/>
    SELECT ?concept ?concept_l ?worksCount
    WHERE {
    """
    patterns = f"""
        ?concept skos:prefLabel ?concept_l . 
        ?concept soap:countsByYear ?countsByYear .
        ?countsByYear soap:year {year} .
        ?countsByYear soap:worksCount ?worksCount .
    """
    concepts_query = query_start + patterns + "}"
    sparql.setReturnFormat(JSON)
    sparql.setQuery(concepts_query)
    all_results = []

    results = sparql.queryAndConvert()
    bindings = results['results']['bindings']
    keys = bindings[0].keys()
    for binding in bindings:
        record = {}
        for key in keys:
            record[key] = binding[key]["value"]
        all_results.append(record)

    return pd.DataFrame(all_results).set_index('concept')


def get_coordinates(input_concept, concepts_list, concepts_embeddings):
    pca = PCA(n_components=2)
    reduced_concepts_embeddings = pca.fit_transform(concepts_embeddings)
    input_index = concepts_list.index(input_concept)
    current_concept_pca = reduced_concepts_embeddings[input_index]

    translated_pca = []
    for reduced_embedding in reduced_concepts_embeddings:
        translated_pca.append(reduced_embedding - current_concept_pca)

    return pd.DataFrame(
        translated_pca,
        columns=["x", "y"],
        index=concepts_list,
    )


def get_concepts_df(input_concept, concepts_list, concepts_embeddings, year, k=10):
    pca = PCA(n_components=2)
    reduced_concepts_embeddings = pca.fit_transform(concepts_embeddings)
    input_index = concepts_list.index(input_concept)
    current_concept_pca = reduced_concepts_embeddings[input_index]

    translated_pca = []
    for reduced_embedding in reduced_concepts_embeddings:
        translated_pca.append(reduced_embedding - current_concept_pca)

    coordinates_df = get_coordinates(input_concept, concepts_list, concepts_embeddings)
    labels_df = get_concept_labels(year)
    labels_df['worksCount'] = labels_df['worksCount'].astype(int)
    neighbors = get_k_nearest_neighbors(input_concept, concepts_list, concepts_embeddings, k)

    neighbors_labels = pd.merge(neighbors, labels_df, left_index=True, right_index=True)

    return pd.merge(neighbors_labels, coordinates_df, left_index=True, right_index=True)


def get_results(
    concept_id,
    year,
    limit,
    max_depth=6,
    max_walks=12,
    with_reverse=True,
    random_seed=42,
    k=10,
):
    g = get_kg(concept_id, year, limit)
    concepts, concepts_embeddings = get_embeddings(g, max_depth, max_walks, with_reverse, random_seed)
    if concepts:
        input_concept = f'https://semopenalex.org/concept/{concept_id}'
        concepts_df = get_concepts_df(input_concept, concepts, concepts_embeddings, year, k)

        return concepts_df.reset_index().to_dict(orient='records')

    else:
        return []
