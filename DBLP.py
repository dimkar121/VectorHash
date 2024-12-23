import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
# from sklearn.metrics.pairwise import cosine_similarity
from CosineLSH import LSH


def embed_sentences_with_distilbert(sentences, model_name="distilbert-base-uncased"):
    """
    Embed sentences using DistilBERT and return vectors.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    # Pass inputs through the model
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the mean pooling of the last hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


def embed_sentences_with_sgtr_t5(sentences, model_name="sentence-transformers/sentence-t5-base"):
    """
    Embed sentences using S-GTR-T5 from the sentence-transformers library.
    """
    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Embed sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)

    return embeddings


'''
def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    """
    # Convert embeddings to numpy arrays
    embedding1 = embedding1.cpu().numpy() if isinstance(embedding1, torch.Tensor) else embedding1
    embedding2 = embedding2.cpu().numpy() if isinstance(embedding2, torch.Tensor) else embedding2

    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity
'''

if __name__ == '__main__':
    df1 = pd.read_csv("../DynaHash/data/DBLP2.csv", sep=",", encoding="utf-8", keep_default_na=False)
    df2 = pd.read_csv("../DynaHash/data/Scholar.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truth = pd.read_csv("../DynaHash/data/truth_Scholar_DBLP.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idDBLP = r["idDBLP"]
        idScholar = r["idScholar"]
        if idDBLP in truthD:
            ids = truthD[idDBLP]
            ids.append(idScholar)
            a += 1
        else:
            truthD[idDBLP] = [idScholar]
    matches = len(truthD.keys()) + a

    tp = 0
    fp = 0
    lsh = LSH(30, 768, .9, .9)

    n = 0
    for i1, r1 in df1.iterrows():
        idDBLP = r1["id"]
        # print(i1)
        authors1 = r1["authors"]
        title1 = r1["title"]
        venue1 = r1["venue"]
        year1 = str(r1["year"])
        # dh.add(title.lower()+" "+authors.lower(), id)
        # Scholars
        distilbert_embeddings1 = embed_sentences_with_distilbert(title1.lower() + " " + authors1.lower())
        lsh.add(title1.lower() + " " + authors1.lower(), distilbert_embeddings1[0], idDBLP)
        # if n == 1000:
        #    break
        n += 1

    for i2, r2 in df2.iterrows():
        authors2 = r2["authors"]
        title2 = r2["title"]
        venue2 = r2["venue"]
        id2 = r2["id"]
        year2 = str(r2["year"])
        # results, _, qtime= dh.get(title.lower()+" "+authors.lower())
        distilbert_embeddings2 = embed_sentences_with_distilbert(title2.lower() + " " + authors2.lower())
        results, n = lsh.get(distilbert_embeddings2[0])
        print("Query used", n, " records in the hash tables.")
        for result in results:
            idDBLP = result["v"]
            if idDBLP in truthD.keys():
                idScholars = truthD[idDBLP]
                for idScholar in idScholars:
                    if idScholar == id2:
                        # print(title2.lower() + " " + authors2.lower())
                        tp += 1
                        print("A TP found.", tp, id2)
                        print(result)
                    else:
                        fp += 1
            else:
                fp += 1
    # end = time.time()
    # print("Total Vectorization time=", end - start, "for", i, "records")
    # print("Avg Vectorization time", (end - start) / i, "seconds")

    print(tp, fp)
    print("recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2))
