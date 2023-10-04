import spacy
from spacy.tokens import Doc
from wasabi import msg

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load("en_coreference_web_trf")

# Example1
doc = nlp(
    "The BRCA genes are known to be associated with breast and ovarian cancers. Mutations within these genes can significantly elevate cancer risks.")

# Example2
# doc = nlp("Alzheimer’s disease is intricately linked to the APOE gene, with the APOE4 variant significantly increasing the risk of developing this debilitating condition.")

# Example3
# doc = nlp("The CFTR gene mutation is a well-known genetic determinant in cystic fibrosis, leading to the production of thick and sticky mucus affecting the lungs and digestive system. Advances in gene therapy are offering hope for correcting this defective gene.")

# Example4
# doc = nlp("Researchers discovered a new link between the PSEN1 gene and early-onset Alzheimer’s. They believe mutations in it exacerbate the disease's progression. A novel compound, CPD-45, has been identified to counteract these effects, and initial studies show that it significantly slows cognitive decline.")

# Print out component names
msg.info("Pipeline components")
for i, pipe in enumerate(nlp.pipe_names):
    print(f"{i}: {pipe}")

# Print out clusters
msg.info("Found clusters")
for cluster in doc.spans:
    print(f"{cluster}: {doc.spans[cluster]}")


def resolve_references(doc: Doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


coref_abs = msg.info("Document with resolved references")
print(resolve_references(doc))