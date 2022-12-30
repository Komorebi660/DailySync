import sys
import numpy as np

sys.path.append('/SPTAG/Release/')
import SPTAG

vnumber = 2000
vdimension = 768


def transform(x):
    norms = np.linalg.norm(x, axis=1)**2
    #print(norms)
    phi = norms.max()
    #print(phi)
    extracol = np.sqrt(phi - norms)
    #print(extracol)
    return np.hstack((extracol.reshape(-1, 1), x)).astype(np.float32)


def build_index():
    np.random.seed(0)
    x = np.random.rand(vnumber, vdimension).astype(np.float32)
    x = transform(x)
    vector_number, vector_dimension = x.shape
    print(vector_number, vector_dimension)

    index = SPTAG.AnnIndex('SPANN', 'Float', vector_dimension)

    # Set the thread number to speed up the build procedure in parallel
    index.SetBuildParam("IndexAlgoType", "BKT", "Base")
    index.SetBuildParam("IndexDirectory", "msmarco", "Base")
    index.SetBuildParam("DistCalcMethod", "L2", "Base")

    index.SetBuildParam("isExecute", "true", "SelectHead")
    index.SetBuildParam("TreeNumber", "1", "SelectHead")
    index.SetBuildParam("BKTKmeansK", "32", "SelectHead")
    index.SetBuildParam("BKTLeafSize", "8", "SelectHead")
    index.SetBuildParam("SamplesNumber", "10000", "SelectHead")
    index.SetBuildParam("SelectThreshold", "50", "SelectHead")
    index.SetBuildParam("SplitFactor", "6", "SelectHead")
    index.SetBuildParam("SplitThreshold", "100", "SelectHead")
    index.SetBuildParam("Ratio", "0.1", "SelectHead")
    index.SetBuildParam("NumberOfThreads", "32", "SelectHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "SelectHead")

    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
    index.SetBuildParam("TPTNumber", "64", "BuildHead")
    index.SetBuildParam("TPTLeafSize", "2000", "BuildHead")
    index.SetBuildParam("MaxCheck", "8192", "BuildHead")
    index.SetBuildParam("MaxCheckForRefineGraph", "8192", "BuildHead")
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", "32", "BuildHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "BuildHead")

    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", "96", "BuildSSDIndex")
    index.SetBuildParam("NumberOfThreads", "32", "BuildSSDIndex")
    index.SetBuildParam("MaxCheck", "8192", "BuildSSDIndex")

    index.SetBuildParam("MaxCheck", "8192", "SearchSSDIndex")
    index.SetBuildParam("NumberOfThreads", "1", "SearchSSDIndex")
    index.SetBuildParam("SearchPostingPageLimit", "96", "SearchSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "64", "SearchSSDIndex")
    index.SetBuildParam("MaxDistRatio", "1000.0", "SearchSSDIndex")

    if index.Build(x, vector_number, False):
        index.Save("test")  # Save the index to the disk


def query():
    index = SPTAG.AnnIndex.Load('test')

    np.random.seed(0)
    q = np.random.rand(vdimension).astype(np.float32)
    q = np.hstack((0.0, q)).astype(np.float32)
    result = index.Search(q, 10)  # Search k=10 nearest vectors for query vector q

    print(result[0])
    print(result[1])


if __name__ == '__main__':
    build_index()
    #exit and run next function
    #query()


'''
[106, 679, 1153, 923, 150, 1596, 1795, 242, 1167, 977]
[132.94873046875, 134.3843536376953, 136.11997985839844, 137.4020538330078, 137.95367431640625, 137.9910430908203, 138.24249267578125, 138.58367919921875, 138.98255920410156, 139.3143768310547]
'''
