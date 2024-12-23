import pandas as pd
import numpy as np
import faiss

if __name__ == '__main__':
    df11 = pd.read_parquet("./data/Abt_embedded_dbert.pqt")
    df12 = pd.read_parquet("./data/Abt_embedded_t5.pqt")

    df21 = pd.read_parquet(
        "./data/Buy_embedded_dbert.pqt")  # , sep=",", encoding="unicode_escape", keep_default_na=False)
    df22 = pd.read_parquet("./data/Buy_embedded_t5.pqt")  # , sep=",", encoding="unicode_escape", keep_default_na=False)

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

    # M = 8  # Number of sub-vectors
    # nlist = 100  # Number of cells in the index
    # Create the quantizer
    # quantizer = faiss.IndexFlatL2(768)  # Flat index for the quantizer (using L2 distance)
    # index = faiss.IndexIVFPQ(quantizer, 768, nlist, M, 8)  # Using PQ with 8 bits per sub-vector

    index = faiss.IndexHNSWFlat(768, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16

    n = 0
    num_rows = df11.shape[0]
    data = np.zeros((num_rows, 768), dtype='float32')
    ids = []

    vectors = []
    for i1, r1 in df11.iterrows():
        id = r1["id"]
        # print(i1)
        name = r1["name"]
        description = r1["description"]
        de11 = r1["v"]  # embed_sentences_with_distilbert(name.lower()+" "+description.lower())
        # data[i1] = de1[0].numpy()
        de12 = df12.loc[i1, "v"]
        np0 = np.stack((de11, de12))
        np00 = np.mean(np0, axis=0)
        vectors.append(np00)
        ids.append([id, name, description])
        n += 1
        # if n >= 10:
        #    break

    # vectors = df11['v'].tolist()
    data = np.array(vectors)
    data = data.astype(np.float32)

    # Step 3: Train the index
    # index.train(data)  # Train on the database vectors

    index.add(data)  # Reshape to 2D array

    for i2, r2 in df21.iterrows():
        name = r2["name"]
        description = r2["description"]
        id2 = r2["id"]
        de21 = np.array(r2["v"])  # embed_sentences_with_distilbert(name.lower() + " " + description.lower())
        de22 = np.array(df22.loc[i2, "v"])
        np0 = np.stack((de21, de22))
        np00 = np.array([np.mean(np0, axis=0)])

        k = 5  # Number of neighbors to retrieve
        distances, indices = index.search(np00, k)

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
