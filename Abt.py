import pandas as pd
import numpy as np
import faiss

if __name__ == '__main__':
    df1 = pd.read_parquet("./data/Abt_embedded_mini.pqt")
    df2 = pd.read_parquet(
        "./data/Buy_embedded_mini.pqt")  # , sep=",", encoding="unicode_escape", keep_default_na=False)
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

    tp = 0
    fp = 0
    d = 384  # Mini  #d=768
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
        de1 = r1["v"]  # embed_sentences_with_distilbert(name.lower()+" "+description.lower())
        # data[i1] = de1[0].numpy()
        ids.append([id, name, description])
        n += 1
        # if n >= 10:
        #    break

    vectors = df1['v'].tolist()
    data = np.array(vectors)
    data = data.astype(np.float32)

    print(data.shape)
    index.add(data)  # Reshape to 2D array

    for i2, r2 in df2.iterrows():
        name = r2["name"]
        description = r2["description"]
        id2 = r2["id"]
        de2 = np.array([r2["v"]])  # embed_sentences_with_distilbert(name.lower() + " " + description.lower())
        k = 5  # Number of neighbors to retrieve
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

    print(tp, fp)
    print("recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2))
