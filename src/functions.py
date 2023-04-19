import rdflib
from SPARQLWrapper import SPARQLWrapper, TURTLE, JSON
import pandas as pd
from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.samplers import PageRankSampler
from pyrdf2vec.walkers import RandomWalker
from sklearn.decomposition import PCA


def get_kg(broader_concept_id, year, limit=500000):
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
        ?work1 soap:hasConcept <https://semopenalex.org/concept/{broader_concept_id}> .  
        ?work1 <http://purl.org/spar/fabio/hasPublicationYear> {year} .
        ?work1 <http://purl.org/spar/cito/cites> ?work2 .
        ?work2 soap:hasConcept <https://semopenalex.org/concept/{broader_concept_id}> .
        ?work1 soap:hasConcept ?topic1 .
        ?work1 <http://purl.org/dc/terms/creator> ?author1 .
        ?work1 soap:hasHostVenue ?hostVenue1 .
        ?hostVenue1 soap:hasVenue ?venue1 .
        ?work2 soap:hasConcept ?topic2 .
    """
    where_end = """
    }
    LIMIT 
    """
    construct_query = query_prefix + where_start + where_patterns + where_end + str(limit)

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
    ttl_path = "src/temp/triples.ttl"
    with open(ttl_path, "w", encoding="utf-8") as file:
        file.write(g.serialize(format='turtle'))
    print('done 1')
    knowledge_graph = KG(
        ttl_path,
        skip_predicates={
            'http://purl.org/spar/fabio/hasPublicationYear',
        },
    )
    print('done 2')
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
    print('done 3')
    walks = transformer.get_walks(
        knowledge_graph,
        concepts,
    )
    print('done 4')
    transformer.fit(walks)
    print('done 5')
    concepts_embeddings, literals = transformer.transform(
        kg=knowledge_graph,
        entities=concepts,
    )

    return concepts_embeddings


def get_concept_labels():
    url = "https://semopenalex.org/sparql"
    sparql = SPARQLWrapper(url)

    concepts_query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT *
    WHERE {?concept skos:prefLabel ?concept_l}
    """

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


def get_concepts_df(concepts_embeddings, concepts, g):
    pca = PCA(n_components=2)
    reduced_concepts_embeddings = pca.fit_transform(concepts_embeddings)

    concepts_df = pd.DataFrame(
        reduced_concepts_embeddings,
        columns=["PC 0", "PC 1"],
        index=concepts,
    )

    has_concept = rdflib.term.URIRef('https://semopenalex.org/property/hasConcept')

    all_works = []
    all_concepts = []
    for work, rdf_predicate, concept in g.triples((None, has_concept, None)):
        all_works.append(str(work))
        all_concepts.append(str(concept))

    concept_works = pd.DataFrame({
        "work": all_works,
        "concept": all_concepts,
    })

    concept_labels = get_concept_labels()
    works_per_concept = concept_works.groupby(by=['concept']).count()
    works_per_concept['log_work'] = works_per_concept['work'].apply(math.log2)
    works_per_concept['label'] = concept_labels['concept_l']

    works_per_concept = pd.merge(
        concepts_df,
        works_per_concept,
        left_index=True,
        right_index=True,
    )

    return works_per_concept
