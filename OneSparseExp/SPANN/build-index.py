import sys
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--passage-path-prefix', type=str, default="../embedding_data/corpus/split0",
                    help='prefix of path to passage embeddings')
parser.add_argument('--n', type=int, default=841823,
                    help='number of inserted data')
args = parser.parse_args()

# add SPTAG Release folder to path
sys.path.append('./SPTAG/Release/')
import SPTAG

corpus = []
for i in range(10):
    with open(args.passage_path_prefix + "%d.pt" % i, "rb") as f:
        embedding, _ = pickle.load(f)
        corpus.append(embedding)
    f.close()
    print("finish %d" % i)

corpus = np.vstack(corpus)
corpus = corpus[:args.n, :]  # get first n data
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
