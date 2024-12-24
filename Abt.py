import pandas as pd
import numpy as np
import faiss

if __name__ == '__main__':
    models = {"albert": 768,
              "dbert": 768,
              "glove": 300,
              "t5": 768,
              "mini": 384,
              "roberta": 768 }
    for model, d in zip(models.keys(), models.values()):
        df1 = pd.read_parquet(f"./data/Abt_embedded_{model}.pqt")
        df2 = pd.read_parquet(f"./data/Buy_embedded_{model}.pqt")
        truth = pd.read_csv("./data/truth_abt_buy.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
        truthD = dict()
        a = 0
        for i, r in truth.iterrows():
            idAbt = r["idAbt"]
            idBuy = r["idBuy"]
            if idAbt in truthD:
                ids = truthD[idAbt]
                ids.append(idBuy)
                a += 1
            else:
                truthD[idAbt] = [idBuy]
        matches = len(truthD.keys()) + a

        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 60
        index.hnsw.efSearch = 16

        n = 0
        num_rows = df1.shape[0]
        data = np.zeros((num_rows, d), dtype='float32')
        ids = []

        for i1, r1 in df1.iterrows():
            id = r1["id"]
            # print(i1)
            name = r1["name"]
            description = r1["description"]
            de1 = r1["v"]
            ids.append([id, name, description])
            n += 1
            # if n >= 10:
            #    break

        vectors = df1['v'].tolist()
        data = np.array(vectors)
        data = data.astype(np.float32)

        index.add(data)

        for k in [1,2,4,6,8]:
            tp = 0
            fp = 0
            for i2, r2 in df2.iterrows():
                name = r2["name"]
                description = r2["description"]
                id2 = r2["id"]
                de2 = np.array([r2["v"]])
                distances, indices = index.search(de2, k)

                for ind in indices[0]:
                    idAbt = ids[ind][0]
                    if idAbt in truthD.keys():
                        idBuys = truthD[idAbt]
                        for idBuy in idBuys:
                            if idBuy == id2:
                                # print(title2.lower() + " " + authors2.lower())
                                tp += 1
                                # print("A TP found.", tp, id2)
                                # print(result)
                            else:
                                fp += 1
                    else:
                        fp += 1

            # end = time.time()
            # print("Total Vectorization time=", end - start, "for", i, "records")
            # print("Avg Vectorization time", (end - start) / i, "seconds")

            print(f"{model} k={k} recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2))
