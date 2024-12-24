import pandas as pd
import numpy as np
import faiss

if __name__ == '__main__':
    models = {"albert": 768,
              "dbert": 768,
              "glove": 300,
              "t5": 768,
              "mini": 384,
              "roberta": 768}
    for model, d in zip(models.keys(), models.values()):
        df1 = pd.read_parquet(f"./data/DBLP_embedded_{model}.pqt")
        df2 = pd.read_parquet(f"./data/ACM_embedded_{model}.pqt")
        truth = pd.read_csv("./data/truth_ACM_DBLP.csv", sep=",", encoding="utf-8", keep_default_na=False)
        truthD = dict()
        a = 0
        for i, r in truth.iterrows():
            idDBLP = r["idDBLP"]
            idACM = r["idACM"]
            truthD[idDBLP] = idACM

        matches = len(truthD.keys())

        n = 0
        num_rows = df1.shape[0]
        data = np.zeros((num_rows, d), dtype='float32')
        ids = []
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 60
        index.hnsw.efSearch = 16

        for i1, r1 in df1.iterrows():
            id = r1["id"]
            authors1 = r1["authors"]
            title1 = r1["title"]
            venue1 = r1["venue"]
            de1 = r1["v"]
            ids.append([id, authors1, title1, venue1])

        vectors = df1['v'].tolist()
        data = np.array(vectors)
        data = data.astype(np.float32)

        index.add(data)  # Reshape to 2D array

        for k in [1,2,4,6,8]:
            tp = 0
            fp = 0
            for i2, r2 in df2.iterrows():
                authors2 = r2["authors"]
                title2 = r2["title"]
                venue2 = r2["venue"]
                id2 = r2["id"]
                de2 = np.array([r2["v"]])
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

            print(f"{model} k={k} recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2))
