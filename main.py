import spacy
from spacy.tokens import Doc
from wasabi import msg
import time


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


def process_whole(pipeline, document_text):
    document = pipeline(document_text)
    with open('whole_document_coref.txt', 'w') as outfile:
        x = outfile.write(f"Found {len(document.spans)} clusters\n")
        for cluster in document.spans:
            x = outfile.write(f"{cluster}: {document.spans[cluster]}\n")
            for span in document.spans[cluster]:
                x = outfile.write(f"({span.start}, {span.end}) {span}\n")
    with open('whole_document_resolved.txt', 'w') as outfile:
        x = outfile.write(resolve_references(document))


def process_by_parts(pipeline, document_text, max_chunk_size=500, delimiter=None):
    start_index = 0
    end_index = 0
    clear_file = open('document_by_parts_resolved.txt', 'w')
    clear_file.close()
    with open('document_by_parts_coref.txt','w') as outfile:
        while start_index + end_index < len(document_text) - 1:
            end_index = start_index + get_substring_index(document_text[start_index:],
                                                          token_count=max_chunk_size,
                                                          token_delimiter=delimiter)
            # print(f"{start_index}, {end_index}")
            subdocument = document_text[start_index:end_index]
            doc = pipeline(subdocument)
            start_index = end_index

            # Print out clusters
            x = outfile.write(f"Found {len(doc.spans)} clusters\n")
            for cluster in doc.spans:
                x = outfile.write(f"{cluster}: {doc.spans[cluster]}\n")
                for span in doc.spans[cluster]:
                    x = outfile.write(f"({span.start}, {span.end}) {span}\n")
            with open('document_by_parts_resolved.txt', 'a') as full_outfile:
                x = full_outfile.write(resolve_references(doc))

            # coref_abs = msg.info("Document with resolved references")
            # print(resolve_references(doc))


def get_substring_index(full_string, token_count=800, token_delimiter=None):
    if token_delimiter:
        tokens = full_string.split(token_delimiter)
    else:
        tokens = full_string.split()
    limit_token_index = len(full_string)
    if len(tokens) > token_count:
        limit_token = tokens[token_count]
        limit_token_index = sum([len(token) for token in tokens[:token_count]]) + token_count
        limit_token_index = full_string.find(limit_token, limit_token_index - 1) + len(limit_token)
        # print(limit_token)
    return limit_token_index


start = time.time()
# spacy.prefer_gpu()

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load("en_coreference_web_trf")

# Example1
# doc = nlp(
#     "The BRCA genes are known to be associated with breast and ovarian cancers. Mutations within these genes can significantly elevate cancer risks.")

# Example2
# doc = nlp("Alzheimer’s disease is intricately linked to the APOE gene, with the APOE4 variant significantly increasing the risk of developing this debilitating condition.")

# Example3
# doc = nlp("The CFTR gene mutation is a well-known genetic determinant in cystic fibrosis, leading to the production of thick and sticky mucus affecting the lungs and digestive system. Advances in gene therapy are offering hope for correcting this defective gene.")

# Example4
# doc = nlp("Researchers discovered a new link between the PSEN1 gene and early-onset Alzheimer’s. They believe mutations in it exacerbate the disease's progression. A novel compound, CPD-45, has been identified to counteract these effects, and initial studies show that it significantly slows cognitive decline.")
text = open('D:\\work\\CRAFT\\11319941.txt', 'r').read()
print(len(text))
process_whole(nlp, text)
end = time.time()
print(f"Completed whole document in {end - start}s")
start = time.time()
nlp = spacy.load("en_coreference_web_trf")
text = open('D:\\work\\CRAFT\\11319941.txt', 'r').read()
print(len(text))
process_by_parts(nlp, text, max_chunk_size=30, delimiter='.')
end = time.time()
print(f"Completed document by parts in {end - start}s")

