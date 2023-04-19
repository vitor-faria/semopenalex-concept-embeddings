import rdflib
from SPARQLWrapper import SPARQLWrapper, TURTLE, JSON
import pandas as pd
from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.samplers import PageRankSampler
from pyrdf2vec.walkers import RandomWalker
from sklearn.decomposition import PCA


def get_kg(broader_concept, year, limit=500000):
    url = "https://semopenalex.org/sparql"
    sparql = SPARQLWrapper(url)
    query_prefix = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX foaf:<http://xmlns.com/foaf/0.1/>
    """
    query_start = """
    CONSTRUCT
    WHERE {
        ?work1 <https://semopenalex.org/property/hasConcept> ?topic1 .
        ?work1 <http://purl.org/spar/cito/cites> ?work2 .
        ?work2 <https://semopenalex.org/property/hasConcept> ?topic2 .
        ?work1 <https://semopenalex.org/property/citedByCount> ?citations1 .
        ?work2 <https://semopenalex.org/property/citedByCount> ?citations2 .
    """
    query_middle = f"""
        ?topic1 skos:broader <https://semopenalex.org/concept/{broader_concept}> .
        ?topic2 skos:broader <https://semopenalex.org/concept/{broader_concept}> .
        ?work1 <http://purl.org/spar/fabio/hasPublicationYear> {year}.
    """
    query_end = """
    }
    LIMIT 
    """
    construct_query = query_prefix + query_start + query_middle + query_end + str(limit)

    sparql.setQuery(construct_query)
    sparql.setReturnFormat(TURTLE)
    results = sparql.query().convert()
    g = rdflib.Graph()
    g.parse(data=results, format="turtle")

    return g


def get_entities(g, broader_concept, min_citation_count=500):
    concepts = set()

    for rdf_subject, rdf_predicate, rdf_object in g.triples((None, None, None)):
        # Subjects
        if '/concept/' in str(rdf_subject) and broader_concept not in str(rdf_subject):
            concepts.add(str(rdf_subject))
        # Objects
        elif '/concept/' in str(rdf_object) and broader_concept not in str(rdf_object):
            concepts.add(str(rdf_object))

    concepts_list = [concept for concept in concepts]

    relevant_works = set()
    citedByCount = rdflib.term.URIRef('https://semopenalex.org/property/citedByCount')
    for subject_work, predicate_cited_count, object_count_citations in g.triples((None, citedByCount, None)):
        if int(object_count_citations) >= min_citation_count:
            relevant_works.add(str(subject_work))

    relevant_works_list = [work for work in relevant_works]

    return relevant_works_list, concepts_list


def get_embeddings(g, works, concepts, year, max_depth=6, max_walks=12, random_seed=42):
    ttl_path = f"{year}_triples.ttl"
    with open(ttl_path, "w", encoding="utf-8") as file:
        file.write(g.serialize(format='turtle'))

    knowledge_graph = KG(
        ttl_path,
        skip_predicates={
            'https://semopenalex.org/property/citedByCount',
            'http://www.w3.org/2004/02/skos/core#broader',
            'http://purl.org/spar/fabio/hasPublicationYear',
        },
    )

    transformer = RDF2VecTransformer(
        Word2Vec(
            seed=random_seed,
            workers=1,
        ),
        walkers=[RandomWalker(
            max_depth=6,
            max_walks=12,
            sampler=PageRankSampler(),
            with_reverse=True,
            md5_bytes=None,
            random_state=random_seed,
        )],
        verbose=2
    )
    walks = transformer.get_walks(
        knowledge_graph,
        works + concepts
    )

    transformer.fit(walks)

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
