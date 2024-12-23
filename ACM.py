import pandas as pd
import numpy as np
import faiss

if __name__ == '__main__':
    df1 = pd.read_parquet("./data/DBLP_embedded_t5.pqt")
    df2 = pd.read_parquet("./data/ACM_embedded_t5.pqt")
    truth = pd.read_csv("./data/truth_ACM_DBLP.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idDBLP = r["idDBLP"]
        idACM = r["idACM"]
        truthD[idDBLP] = idACM

    matches = len(truthD.keys())
    print("Truth=", matches)

    tp = 0
    fp = 0
    n = 0
    n = 0
    num_rows = df1.shape[0]
    data = np.zeros((num_rows, 768), dtype='float32')
    ids = []
    index = faiss.IndexHNSWFlat(768, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16

    for i1, r1 in df1.iterrows():
        id = r1["id"]
        authors1 = r1["author"]
        title1 = r1["title"]
        venue1 = r1["venue"]
        de1 = r1["v"]  # embed_sentences_with_distilbert(name.lower()+" "+description.lower())
        ids.append([id, authors1, title1, venue1])

    vectors = df1['v'].tolist()
    data = np.array(vectors)
    data = data.astype(np.float32)

    index.add(data)  # Reshape to 2D array

    for i2, r2 in df2.iterrows():
        authors2 = r2["authors"]
        title2 = r2["title"]
        venue2 = r2["venue"]
        id2 = r2["id"]
        de2 = np.array([r2["v"]])  # embed_sentences_with_distilbert(name.lower() + " " + description.lower())
        k = 1  # Number of neighbors to retrieve
        distances, indices = index.search(de2, k)

        for ind in indices[0]:
            idDBLP = ids[ind][0]
            if idDBLP in truthD.keys():
                idACM = truthD[idDBLP]
                if idACM == id2:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1

    print(tp, fp)
    print("recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2))
