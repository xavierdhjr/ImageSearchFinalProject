#include <stdexcept>
#include <vector>
#include "flann.h"
#include "util/Timer.h"
#include "util/common.h"
#include "util/Logger.h"
#include "algorithms/KDTree.h"
#include "algorithms/KMeansTree.h"
#include "algorithms/CompositeTree.h"
#include "algorithms/LinearSearch.h"
#include "nn/Autotune.h"
#include "nn/Testing.h"
#include <objbase.h>
using namespace std;


#include <iostream>
#include <fstream>
#include "flann.h"

#ifdef WIN32
#define EXPORTED extern "C" __declspec(dllexport)
#else
#define EXPORTED extern "C"
#endif


namespace {

    typedef NNIndex* NNIndexPtr;
    typedef Dataset<float>* DatasetPtr;
    
    const char* algos[] = { "linear","kdtree", "kmeans", "composite" };
    const char* centers_algos[] = { "random", "gonzales", "kmeanspp" };

	
	Params parametersToParams(IndexParameters parameters)
	{
		Params p;
		p["checks"] = parameters.checks;
        p["cb_index"] = parameters.cb_index;
		p["trees"] = parameters.trees;
		p["max-iterations"] = parameters.iterations;
		p["branching"] = parameters.branching;
		p["target-precision"] = parameters.target_precision;
		
		if (parameters.centers_init >=0 && parameters.centers_init<ARRAY_LEN(centers_algos)) {
			p["centers-init"] = centers_algos[parameters.centers_init];
		}
		else {
			p["centers-init"] = "random";
		}
		
		if (parameters.algorithm >=0 && parameters.algorithm<ARRAY_LEN(algos)) {
			p["algorithm"] = algos[parameters.algorithm];
		}

		return p;
	}
	
	IndexParameters paramsToParameters(Params params)
	{
		IndexParameters p;
		
		try {
			p.checks = (int)params["checks"];
		} catch (...) {
			p.checks = -1;
		}

        try {
            p.cb_index = (float)params["cb_index"];
        } catch (...) {
            p.cb_index = 0.4;
        }
        
		try {
			p.trees = (int)params["trees"];
		} catch (...) {
			p.trees = -1;
		}

		try {
			p.iterations = (int)params["max-iterations"];
		} catch (...) {
			p.iterations = -1;
		}
		try {
			p.branching = (int)params["branching"];
		} catch (...) {
			p.branching = -1;
		}
		try {
  			p.target_precision = (float)params["target-precision"];
		} catch (...) {
			p.target_precision = -1;
		}
        p.centers_init = CENTERS_RANDOM;
        for (size_t algo_id =0; algo_id<ARRAY_LEN(centers_algos); ++algo_id) {
            const char* algo = centers_algos[algo_id];
            try {
				if (algo == params["centers-init"] ) {
					p.centers_init = algo_id;
					break;
				}
			} catch (...) {}
		}
        p.algorithm = LINEAR;
        for (size_t algo_id =0; algo_id<ARRAY_LEN(algos); ++algo_id) {
            const char* algo = algos[algo_id];
			if (algo == params["algorithm"] ) {
				p.algorithm = algo_id;
				break;
			}
		}
		return p;
	}
}



void init_flann_parameters(FLANNParameters* p)
{
	//printf("flann params?\n");
	if (p != NULL) {
 		flann_log_verbosity(p->log_level);
		//printf("Set flann log verbosity\n");
		flann_log_destination(p->log_destination);
		//printf("Set flann log destination\n");
        if (p->random_seed>0) {
		  seed_random(p->random_seed);
        }
		//printf("Got random seed\n");
	}
	//printf("yup\n");
}

void readSizes(vector<int>* outSizes)
{
	const char SIZES_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\features\\esp.size";
	ifstream sizeFileStream;
	sizeFileStream.open(SIZES_FILE);
	
	if(sizeFileStream.is_open())
	{
		cout << "Successfully opened " << SIZES_FILE << endl;

		int i = 0;
		while(!sizeFileStream.eof())
		{
			if(i > 50) // arbitrary limit to reduce time
				break;
			int size;
			sizeFileStream >> size;
			(*outSizes).push_back(size);
			++i;
		}

		cout << "Pushed back " << (*outSizes).size() << " sizes." << endl;
	}
	else
	{
		cout << "There was a problem opening " << SIZES_FILE << endl;
		return;
	}
}

float* readFeatures(int total_keypoints, const int KEYPOINT_SIZE)
{
	const char FEATURE_FILE[] ="C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\features\\esp.feature";
	ifstream featureFileStream;
	featureFileStream.open(FEATURE_FILE);
	
	float* data = new float[total_keypoints * KEYPOINT_SIZE];
	if(featureFileStream.is_open())
	{
		cout << "Successfully opened " << FEATURE_FILE << endl;

		int current_keypoint = 0;
		while(current_keypoint < total_keypoints)
		{
			int k = 0;
			float n;
			while(k < KEYPOINT_SIZE)
			{
				featureFileStream >> n;
				data[current_keypoint * KEYPOINT_SIZE + k] = n;
				++k;
			}

			current_keypoint++;
		}
		
		cout << "Read in " << current_keypoint << " keypoints." << endl;
		featureFileStream.close();
	}
	else
	{
		cout << "There was a problem opening " << FEATURE_FILE << endl;
	}
	return data;
}

void readImageNames(vector<string>* outNames)
{
	const char IMAGELIST_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\features\\imglist.txt";
	ifstream imageFileStream;
	imageFileStream.open(IMAGELIST_FILE);
	
	if(imageFileStream.is_open())
	{
		cout << "Successfully opened " << IMAGELIST_FILE << endl;

		int i = 0;
		while(!imageFileStream.eof())
		{
			if(i > 50) // arbitrary limit to reduce time
				break;
			string size;
			imageFileStream >> size;
			(*outNames).push_back(size);
			++i;
		}

		cout << "Pushed back " << (*outNames).size() << " image names." << endl;
	}
	else
	{
		cout << "There was a problem opening " << IMAGELIST_FILE << endl;
		return;
	}
}

void writeClusterData(float* cluster_centers, int clusters_returned, const int KEYPOINT_SIZE)
{
	const char CLUSTER_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\clusters.txt";
	ofstream clusterFileStream;
	clusterFileStream.open(CLUSTER_FILE);
	if(clusterFileStream.is_open())
	{
		cout << "Successfully opened " << CLUSTER_FILE << endl;
		clusterFileStream << clusters_returned << endl;
		clusterFileStream << KEYPOINT_SIZE << endl;

		for(int i = 0; i < clusters_returned; ++i)
		{
			for(int k = 0; k < KEYPOINT_SIZE; ++k)
			{
				clusterFileStream << cluster_centers[i * KEYPOINT_SIZE + k] << " ";
			}
			clusterFileStream << endl;
		}
		cout << "Wrote " << clusters_returned << " cluster centers to file." << endl;
	}
	else
	{
		cout << "Could not open " << CLUSTER_FILE << endl;
	}
}

void buildIndex(FLANN_INDEX* outIndex, float* cluster_centers, int num_clusters, int KEYPOINT_SIZE)
{
	IndexParameters build_index_params;
	build_index_params.algorithm = KDTREE;
	build_index_params.checks = 2048;
	build_index_params.trees = 8;
	build_index_params.target_precision = -1;
	build_index_params.build_weight = 0.01;
	build_index_params.memory_weight = 1;

	float speedup;
	*outIndex = flann_build_index(cluster_centers,num_clusters,KEYPOINT_SIZE,&speedup, &build_index_params,NULL);
	cout << "Build index. " << endl;
}
float* ReadClusterFile(char* clusterOutputFile, int* numClusters)
{
	std::ifstream fileStream;
	fileStream.open(clusterOutputFile);
	int length = 0;
	// Each line has a size on it
	cout << clusterOutputFile << endl;
	int total_point_dims;
	int num_dimensions;
	float* cluster_centers;
	if(fileStream.is_open())
	{


		fileStream >> total_point_dims;
		fileStream >> num_dimensions;
		length = total_point_dims * num_dimensions;
		
		cluster_centers = new float[length];

		int i = 0;
		std::string str;
		float dim = 0;
		cout << "haha" << endl;
		while(i < length)
		{	
			for(int k = 0; k < 100000 && i < length; ++k)
			{

				fileStream >> dim;
				cluster_centers[i] = dim;
				//cout << dim << endl;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;

				fileStream >> dim;
				cluster_centers[i] = dim;
				++i;
			}
			double percent_complete = (double)i / (double)length;
			printf("Read %16f%% of keypoint dimensions (%d / %d).\n", percent_complete * 100, i, length);

		}
	}else
	{
		std::cout << "Error opening file " << clusterOutputFile << std::endl;
		return 0;
	}


	*numClusters = total_point_dims; // set value of numClusters to length
	std::cout << "Finished reading cluster file. Read " << length << " sizes." << std::endl;
	fileStream.close();
	return cluster_centers;
}
/*
EXPORTED char* CreateBagOfWords()
{
    char szSampleString[] = "Hello World";
    ULONG ulSize = strlen(szSampleString) + sizeof(char);
    char* pszReturn = NULL;

    pszReturn = (char*)::CoTaskMemAlloc(ulSize);
    // Copy the contents of szSampleString
    // to the memory pointed to by pszReturn.
    strcpy(pszReturn, szSampleString);
    // Return pszReturn.
    return pszReturn;
}*/
char* sayBeans()
{
	//const char* szSampleString = strStream.str().c_str();
	char szSampleString[] = "beans";
    ULONG ulSize = strlen(szSampleString) + sizeof(char);
    char* pszReturn = NULL;

    pszReturn = (char*)::CoTaskMemAlloc(ulSize);
    // Copy the contents of szSampleString
    // to the memory pointed to by pszReturn.
    strcpy(pszReturn, szSampleString);
    // Return pszReturn.
    return pszReturn;
}

EXPORTED char* CreateBagOfWords(float* keypoint_data, int num_keypoints)
{	


	char CLUSTER_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\clusters.txt";
	int num_clusters;
	//float* cluster_centers = (float*)::CoTaskMemAlloc(4609 * 128 * sizeof(float));
		cout << "ha" << endl;
	float* cluster_centers = ReadClusterFile(CLUSTER_FILE, &num_clusters);

	FLANN_INDEX index;

	buildIndex(&index, cluster_centers, num_clusters, 128);

	stringstream strStream;

	cout << "Writing bag of words. " << endl;

	FLANNParameters flann_params;
	flann_params.log_level = LOG_NONE;
	flann_params.log_destination = NULL;
	flann_params.random_seed = CENTERS_RANDOM;

	strStream << "<DOC>" << endl;
	strStream << "<DOCNO>" << "beans" << "</DOCNO>" << endl;
	strStream << "<TEXT>" << endl;

	int keypoints_examined = 0;
	const int KEYPOINT_SIZE = 128;
	for(int j = 0; j < num_keypoints / KEYPOINT_SIZE; ++j) //(num_keypoints / KEYPOINT_SIZE)
	{
		float* keypoint = new float[KEYPOINT_SIZE];
		for(int k = 0; k < KEYPOINT_SIZE; ++k)
		{
			keypoint[k] = keypoint_data[keypoints_examined * KEYPOINT_SIZE + k];
		}
		keypoints_examined++;
		int* nearest_neighbor = new int[1];
		int _r = flann_find_nearest_neighbors_index(index,keypoint,1,nearest_neighbor,1,1024,&flann_params);
		strStream << "w" << nearest_neighbor[0] << " ";
	}
	strStream << endl << "</TEXT>" << endl;
	strStream << "</DOC>" << endl;

	cout << "Wrote out " << keypoints_examined << " keypoints." << endl;

	const char* szSampleString = strStream.str().c_str();
	cout << strStream.str() << endl;
	cout << strStream.str().c_str() << endl;
    ULONG ulSize = strlen(szSampleString) + sizeof(char);
	cout << strlen(szSampleString) << endl;
    char* pszReturn = NULL;

    pszReturn = (char*)::CoTaskMemAlloc(ulSize);
    // Copy the contents of szSampleString
    // to the memory pointed to by pszReturn.
    strcpy(pszReturn, strStream.str().c_str());
	cout << pszReturn << endl;
    // Return pszReturn.
    return pszReturn;
}

EXPORTED void UpdateClusterCenters(char sizeFile[], char featureFile[], char clusterOutputFile[])
{

	/*
		Read in keypoints
	*/

	const char FEATURE_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\features\\esp.feature";
	const char SIZES_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\features\\esp.size";
	const char CLUSTER_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\clusters.txt";
	const char BAGOWORDS_FILE[] = "C:\\Users\\Raider\\ImageSearchFinalProject\\cse484project\\bagofwords\\bagofwords.txt";

	const int KEYPOINT_SIZE = 128;

	ifstream sizeFileStream;
	sizeFileStream.open(SIZES_FILE);

	vector<int> sizes;
	readSizes(&sizes);

	int total_keypoints = 0;
	for(int i = 0; i < sizes.size(); ++i)
	{
		total_keypoints += sizes[i];
	}

	cout << "There are " << total_keypoints << " keypoints." << endl;

	float* flann_data = readFeatures(total_keypoints, KEYPOINT_SIZE);

	/*
		
		Call FLANN Library

	*/
	const int CLUSTERS = 1500000;
	IndexParameters index_params;
	index_params.algorithm = KMEANS;
	index_params.checks = 2048;
	index_params.cb_index = 0.6;
	index_params.branching = 10;
	index_params.iterations = 15;
	index_params.centers_init = CENTERS_GONZALES;
	index_params.target_precision = -1;
	index_params.build_weight = 0.01;
	index_params.memory_weight = 1;
	float* cluster_centers = new float[CLUSTERS * KEYPOINT_SIZE];
	int flann_result = flann_compute_cluster_centers(flann_data, total_keypoints, KEYPOINT_SIZE, CLUSTERS, cluster_centers, &index_params, NULL);
	cout << "Flann result: " << flann_result << endl;
	
	int clusters_returned = flann_result;
	writeClusterData(cluster_centers, clusters_returned, KEYPOINT_SIZE);

	IndexParameters build_index_params;
	build_index_params.algorithm = KDTREE;
	build_index_params.checks = 2048;
	build_index_params.trees = 8;
	build_index_params.target_precision = -1;
	build_index_params.build_weight = 0.01;
	build_index_params.memory_weight = 1;

	float speedup;
	FLANN_INDEX index = flann_build_index(cluster_centers,clusters_returned,KEYPOINT_SIZE,&speedup, &build_index_params,NULL);
	cout << "Build index. " << endl;

	ofstream bagOfWordsStream;
	bagOfWordsStream.open(BAGOWORDS_FILE);
	vector<string> imgNames;
	readImageNames(&imgNames);

	if(bagOfWordsStream.is_open())
	{
		cout << "Successfully opened " << BAGOWORDS_FILE << endl;

		FLANNParameters flann_params;
		flann_params.log_level = LOG_NONE;
		flann_params.log_destination = NULL;
		flann_params.random_seed = CENTERS_RANDOM;
		int keypoints_examined = 0;
		for(int i = 0; i < sizes.size(); ++i)
		{

			//int* nearest =  &nearest_neighbors_result[i];
			//int nearest_num = (int)(*(nearest + 4)); // the cluster center id is stored in the next 4 bytes
			bagOfWordsStream << "<DOC>" << endl;
			bagOfWordsStream << "<DOCNO>" << imgNames[i] << "</DOCNO>" << endl;
			bagOfWordsStream << "<TEXT>" << endl;

			int keypoints_i = sizes[i];
			for(int j = 0; j < sizes[i]; ++j)
			{
				float* keypoint = new float[KEYPOINT_SIZE];
				for(int k = 0; k < KEYPOINT_SIZE; ++k)
				{
					keypoint[k] = flann_data[keypoints_examined * KEYPOINT_SIZE + k];
				}
				keypoints_examined++;
				int* nearest_neighbor = new int[1];
				int _r = flann_find_nearest_neighbors_index(index,keypoint,1,nearest_neighbor,1,1024,&flann_params);
				bagOfWordsStream << "w" << nearest_neighbor[0] << " ";
			}
			bagOfWordsStream << endl << "</TEXT>" << endl;
			bagOfWordsStream << "</DOC>" << endl;
			
		}
		cout << "Wrote " << sizes.size() << " bag of words documents." << endl;
		bagOfWordsStream.close();
	}
	else
	{
		cout << "There was a problem opening " << BAGOWORDS_FILE << endl;
	}

	
}

EXPORTED void flann_log_verbosity(int level)
{
    if (level>=0) {
        logger.setLevel(level);
    }
}

EXPORTED void flann_log_destination(char* destination)
{
    logger.setDestination(destination);
	//printf("Destination set!\n");
}


EXPORTED FLANN_INDEX flann_build_index(float* dataset, int rows, int cols, float* speedup, IndexParameters* index_params, FLANNParameters* flann_params)
{	
	try {

		init_flann_parameters(flann_params);
		//printf("finished init\n");
		DatasetPtr inputData = new Dataset<float>(rows,cols,dataset);
		//printf("finished input data setup\n");
		if (index_params == NULL) {
			throw FLANNException("The index_params agument must be non-null");
		}

		float target_precision = index_params->target_precision;
        float build_weight = index_params->build_weight;
        float memory_weight = index_params->memory_weight;
        float sample_fraction = index_params->sample_fraction;
		//printf("init'd vars\n");
		NNIndex* index = NULL;
		if (target_precision < 0) {
			Params params = parametersToParams(*index_params);
			logger.info("Building index\n");
			index = create_index((const char *)params["algorithm"],*inputData,params);
            StartStopTimer t;
            t.start();

				//printf("About to build index\n");

            index->buildIndex();

				//printf("built index in %3f second\n", t.value);

            t.stop();
            logger.info("Building index took: %g\n",t.value);
		}
		else {
            if (index_params->build_weight < 0) {
                throw FLANNException("The index_params.build_weight must be positive.");
            }
            
            if (index_params->memory_weight < 0) {
                throw FLANNException("The index_params.memory_weight must be positive.");
            }
            Autotune autotuner(index_params->build_weight, index_params->memory_weight, index_params->sample_fraction);    
			Params params = autotuner.estimateBuildIndexParams(*inputData, target_precision);
			index = create_index((const char *)params["algorithm"],*inputData,params);
			index->buildIndex();
			autotuner.estimateSearchParams(*index,*inputData,target_precision,params);

			*index_params = paramsToParameters(params);
			index_params->target_precision = target_precision;
            index_params->build_weight = build_weight;
            index_params->memory_weight = memory_weight;
            index_params->sample_fraction = sample_fraction;
            
			if (speedup != NULL) {
				*speedup = float(params["speedup"]);
			}
		}

		return index;
	}
	catch (runtime_error& e) {
		logger.error("Caught exception: %s\n",e.what());
		return NULL;
	}
}


EXPORTED int flann_find_nearest_neighbors(float* dataset,  int rows, int cols, float* testset, int tcount, int* result, int nn, IndexParameters* index_params, FLANNParameters* flann_params)
{
	try {
		init_flann_parameters(flann_params);
		
        DatasetPtr inputData = new Dataset<float>(rows,cols,dataset);
		float target_precision = index_params->target_precision;

        StartStopTimer t;
		NNIndexPtr index;
		if (target_precision < 0) {
			Params params = parametersToParams(*index_params);
			logger.info("Building index\n");
            index = create_index((const char *)params["algorithm"],*inputData,params);
            t.start();
 			index->buildIndex();
            t.stop();
            logger.info("Building index took: %g\n",t.value);
		}
		else {
            logger.info("Build index: %g\n", index_params->build_weight);
            Autotune autotuner(index_params->build_weight, index_params->memory_weight, index_params->sample_fraction);    
            Params params = autotuner.estimateBuildIndexParams(*inputData, target_precision);
            index = create_index((const char *)params["algorithm"],*inputData,params);
            index->buildIndex();
            autotuner.estimateSearchParams(*index,*inputData,target_precision,params);
			*index_params = paramsToParameters(params);
		}
		logger.info("Finished creating the index.\n");
		
		logger.info("Searching for nearest neighbors.\n");
        Params searchParams;
        searchParams["checks"] = index_params->checks;
        Dataset<int> result_set(tcount, nn, result);
        search_for_neighbors(*index, Dataset<float>(tcount, cols, testset), result_set, searchParams);
		
		delete index;
		delete inputData;
		
		return 0;
	}
	catch(runtime_error& e) {
		logger.error("Caught exception: %s\n",e.what());
		return -1;
	}
}

EXPORTED int flann_find_nearest_neighbors_index(FLANN_INDEX index_ptr, float* testset, int tcount, int* result, int nn, int checks, FLANNParameters* flann_params)
{
	try {
		init_flann_parameters(flann_params);
        
        if (index_ptr==NULL) {
            throw FLANNException("Invalid index");
        }
        NNIndexPtr index = NNIndexPtr(index_ptr);
		//printf("Setup nnindex ptr\n");
        int length = index->veclen();        
        StartStopTimer t;
        t.start();
        Params searchParams;
        searchParams["checks"] = checks;
        Dataset<int> result_set(tcount, nn, result);
		//printf("Setup result set\n");
        search_for_neighbors(*index, Dataset<float>(tcount, length, testset), result_set, searchParams);
		//printf("nearest neighbor search complete!\n");
        t.stop();
        logger.info("Searching took %g seconds\n",t.value);

		return 0;
	}
	catch(runtime_error& e) {
		logger.error("Caught exception: %s\n",e.what());
		return -1;
	}
	
}

int flann_free_index(FLANN_INDEX index_ptr, FLANNParameters* flann_params)
{
	try {
		init_flann_parameters(flann_params);

        if (index_ptr==NULL) {
            throw FLANNException("Invalid index");
        }
        NNIndexPtr index = NNIndexPtr(index_ptr);
        delete index;
     
        return 0;   
	}
	catch(runtime_error& e) {
		logger.error("Caught exception: %s\n",e.what());
        return -1;
	}
}

EXPORTED int flann_compute_cluster_centers(float* dataset, int rows, int cols, int clusters, float* result, IndexParameters* index_params, FLANNParameters* flann_params)
{
	printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
	try {
		printf("Staring computer cluster centers time.\n");
		init_flann_parameters(flann_params);
		printf("Init'd flann params!\n");
        DatasetPtr inputData = new Dataset<float>(rows,cols,dataset);
		printf("Created dataset!\n");
        Params params = parametersToParams(*index_params);
        KMeansTree kmeans(*inputData, params);
		kmeans.buildIndex();
		printf("built dat index!\n");
        int clusterNum = kmeans.getClusterCenters(clusters,result);
		return clusterNum;
	} catch (runtime_error& e) {
		printf("Sexception!\n");
		logger.error("Caught exception: %s\n",e.what());
		return -1;
	}
}


EXPORTED void compute_ground_truth_float(float* dataset, int dshape[], float* testset, int tshape[], int* match, int mshape[], int skip)
{
    assert(dshape[1]==tshape[1]);
    assert(tshape[0]==mshape[0]);

    Dataset<int> _match(mshape[0], mshape[1], match);
    compute_ground_truth(Dataset<float>(dshape[0], dshape[1], dataset), Dataset<float>(tshape[0], tshape[1], testset), _match, skip);
}


EXPORTED float test_with_precision(FLANN_INDEX index_ptr, float* dataset, int dshape[], float* testset, int tshape[], int* matches, int mshape[],
             int nn, float precision, int* checks, int skip = 0)
{
    assert(dshape[1]==tshape[1]);
    assert(tshape[0]==mshape[0]);

    try {
        if (index_ptr==NULL) {
            throw FLANNException("Invalid index");
        }
        NNIndexPtr index = (NNIndexPtr)index_ptr;
        return test_index_precision(*index, Dataset<float>(dshape[0], dshape[1],dataset), Dataset<float>(tshape[0], tshape[1], testset), 
                Dataset<int>(mshape[0],mshape[1],matches), precision, *checks, nn, skip);
    } catch (runtime_error& e) {
        logger.error("Caught exception: %s\n",e.what());
        return -1;
    }
}

EXPORTED float test_with_checks(FLANN_INDEX index_ptr, float* dataset, int dshape[], float* testset, int tshape[], int* matches, int mshape[],
             int nn, int checks, float* precision, int skip = 0)
{
    assert(dshape[1]==tshape[1]);
    assert(tshape[0]==mshape[0]);

    try {
        if (index_ptr==NULL) {
            throw FLANNException("Invalid index");
        }
        NNIndexPtr index = (NNIndexPtr)index_ptr;
        return test_index_checks(*index, Dataset<float>(dshape[0], dshape[1],dataset), Dataset<float>(tshape[0], tshape[1], testset), 
                Dataset<int>(mshape[0],mshape[1],matches), checks, *precision, nn, skip);
    } catch (runtime_error& e) {
        logger.error("Caught exception: %s\n",e.what());
        return -1;
    }
}
