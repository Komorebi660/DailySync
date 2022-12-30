import sys
import pickle
import numpy as np

sys.path.append('/SPTAG/Release/')
import SPTAG


def transform(x):
    norms = np.linalg.norm(x, axis=1)**2
    phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((extracol.reshape(-1, 1), x)).astype(np.float32)


corpus = []
for i in range(10):
    with open("../../embedding_data/corpus/split0%d.pt" % i, "rb") as f:
        embedding, _ = pickle.load(f)
        corpus.append(embedding)
    f.close()
    print("finish %d" % i)

corpus = np.vstack(corpus)
print("Transform Begin !!!")
corpus = transform(corpus)
print("Transform End !!!")
vector_number, vector_dim = corpus.shape
print(vector_number, vector_dim)


print("Build index Begin !!!")

index = SPTAG.AnnIndex('SPANN', 'Float', vector_dim)

# Set the thread number to speed up the build procedure in parallel
index.SetBuildParam("IndexAlgoType", "BKT", "Base")
index.SetBuildParam("IndexDirectory", "msmarco", "Base")
index.SetBuildParam("DistCalcMethod", "L2", "Base")

index.SetBuildParam("isExecute", "true", "SelectHead")
index.SetBuildParam("TreeNumber", "1", "SelectHead")
index.SetBuildParam("BKTKmeansK", "32", "SelectHead")
index.SetBuildParam("BKTLeafSize", "8", "SelectHead")
index.SetBuildParam("SamplesNumber", "10000", "SelectHead")
#index.SetBuildParam("SaveBKT", "false", "SelectHead")
index.SetBuildParam("SelectThreshold", "50", "SelectHead")
index.SetBuildParam("SplitFactor", "6", "SelectHead")
index.SetBuildParam("SplitThreshold", "100", "SelectHead")
index.SetBuildParam("Ratio", "0.1", "SelectHead")
index.SetBuildParam("NumberOfThreads", "64", "SelectHead")
index.SetBuildParam("BKTLambdaFactor", "-1", "SelectHead")

index.SetBuildParam("isExecute", "true", "BuildHead")
index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
index.SetBuildParam("TPTNumber", "64", "BuildHead")
index.SetBuildParam("TPTLeafSize", "2000", "BuildHead")
index.SetBuildParam("MaxCheck", "8192", "BuildHead")
index.SetBuildParam("MaxCheckForRefineGraph", "8192", "BuildHead")
index.SetBuildParam("RefineIterations", "3", "BuildHead")
index.SetBuildParam("NumberOfThreads", "64", "BuildHead")
index.SetBuildParam("BKTLambdaFactor", "-1", "BuildHead")

index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex")
index.SetBuildParam("PostingPageLimit", "96", "BuildSSDIndex")
index.SetBuildParam("NumberOfThreads", "64", "BuildSSDIndex")
index.SetBuildParam("MaxCheck", "8192", "BuildSSDIndex")

index.SetBuildParam("SearchPostingPageLimit", "96", "BuildSSDIndex")
index.SetBuildParam("SearchInternalResultNum", "64", "BuildSSDIndex")
index.SetBuildParam("MaxDistRatio", "1000.0", "BuildSSDIndex")

index.SetBuildParam("MaxCheck", "8192", "SearchSSDIndex")
index.SetBuildParam("NumberOfThreads", "1", "SearchSSDIndex")
index.SetBuildParam("SearchPostingPageLimit", "96", "SearchSSDIndex")
index.SetBuildParam("SearchInternalResultNum", "64", "SearchSSDIndex")
index.SetBuildParam("MaxDistRatio", "1000.0", "SearchSSDIndex")

if index.Build(corpus, vector_number, False):
    index.Save("msmarco")  # Save the index to the disk

print("Build index Down !!!")
