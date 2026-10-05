from lanobert.ontology import OntologyDocument, TfidfOntologyRetriever


def test_ontology_retriever_returns_relevant_document_first():
    retriever = TfidfOntologyRetriever([
        OntologyDocument("timeout", "scheduler timeout heartbeat exceeded"),
        OntologyDocument("memory", "memory pressure causes allocation failure"),
    ])
    results = retriever.retrieve("heartbeat timeout", top_k=1)
    assert results[0][0].id == "timeout"
    assert results[0][1] > 0
