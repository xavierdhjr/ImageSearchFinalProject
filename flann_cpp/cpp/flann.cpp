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
	const char SIZES_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\cse484project\\features\\esp.size";
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
	const char FEATURE_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\cse484project\\features\\esp.feature";
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
	const char IMAGELIST_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\cse484project\\features\\esp.size";
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
	const char CLUSTER_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\clusters.txt";
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

EXPORTED void CreateBagOfWords(char* str)
{
	//buffer = new char[]{ "bingo" };
	str = new char[20];
	str[0] = 'a';
}

EXPORTED void UpdateClusterCenters(char sizeFile[], char featureFile[], char clusterOutputFile[])
{

	/*
		Read in keypoints
	*/

	const char FEATURE_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\cse484project\\features\\esp.feature";
	const char SIZES_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\cse484project\\features\\esp.size";
	const char CLUSTER_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\clusters.txt";
	const char BAGOWORDS_FILE[] = "C:\\Users\\Raider\\Desktop\\MSU\\FS13\\CSE484\\project\\bagofwords\\bagofwords.txt";

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
			bagOfWordsStream << "<DOCNO>" << i << "</DOCNO>" << endl;
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

// Format of file:
// dimensionality of each row at top
// 
EXPORTED int* FindNearestNeighbors(char* clusterFile, float* imageQuery)
{
	printf("ding dong\n");
	int* result = new int[0];
	float* dataset;
	vector<float> points;
	std::ifstream clusterFileStream;
	clusterFileStream.open(clusterFile);
	int rows = 0;
	int columns = 0;

	if(clusterFileStream.is_open())
	{
		std::string dimensionality;
		clusterFileStream >> dimensionality;
		columns = atoi(dimensionality.c_str());

		int i = 0;
		printf("Reading cluster document with dimensionality %d", columns);
		while(!clusterFileStream.eof())
		{
			std::string str;
			clusterFileStream >> str;
//			int size = atoi(str.c_str());
//			sizes.push_back(size);
			float f = atof(str.c_str());

			points.push_back(f);
			if(i < columns)
			{
				++i;
			}
			else
			{
				++rows;
				i = 0;
			}
		}
		printf("Read %d points. Rows: %d, Cols: %d\n", points.size(), rows, columns);
		dataset = new float[points.size()];
		for(int j = 0; j < points.size(); ++j)
		{
			dataset[j] = points[j];
		}


	}else
	{
		std::cout << "Error opening file " << clusterFile << std::endl;
		return result;
	}
	clusterFileStream.close();
	float speedup;
	
	IndexParameters build_index_params;
	build_index_params.algorithm = KDTREE;
	build_index_params.checks = 2048;
	build_index_params.trees = 8;
	build_index_params.target_precision = -1;
	build_index_params.build_weight = 0.01;
	build_index_params.memory_weight = 1;
	printf("Constructing index\n");
	FLANN_INDEX index = flann_build_index(dataset, rows, columns, &speedup, &build_index_params, NULL);
	printf("Index constructed.\n");

	result = new int[rows];

	FLANNParameters flann_params;
	flann_params.log_level = LOG_NONE;
	flann_params.log_destination = NULL;
	flann_params.random_seed = CENTERS_RANDOM;
	printf("Searching for nearest neighbors\n");
	int r = flann_find_nearest_neighbors_index(index, imageQuery, columns, result, 1, 1024, &flann_params);
	printf("Nearest neighbor search complete. Returned code %d", r);

	return result;
}
/*
void WriteBagOfWords(char* imgListFile, char* bagOfWordsOutputFile, char* featureFile, FLANN_INDEX index, const int KEYPOINT_DIMENSIONALITY)
{
	const char null_byte[] = { '\0' };
		
	std::ofstream bagOfWordsFile;
	//char* outputFile = bagOfWordsOutputFile;
	char* outputFile = new char[strlen(bagOfWordsOutputFile) + 1];

	strcpy(outputFile, bagOfWordsOutputFile);

	std::ifstream imgListFileStream;
	imgListFileStream.open(imgListFile);

	vector<std::string> fileNames;

	printf("Reading in image file names\n");
	if(imgListFileStream.is_open())
	{
		while(!imgListFileStream.eof())
		{
			std::string str;
			imgListFileStream >> str;
			fileNames.push_back(str);
		}
	}else{
		printf("Error opening file %s\n",imgListFile);
		return;
	}

	std::stringstream strstream;
	strstream << "bagofwords.txt";
	const char* doc_num = strstream.str().c_str();
	std::string beans(outputFile);
	std::string backslash("\\");
	std::string outputFileForReal = outputFile + backslash + strstream.str();
	std::cout << outputFileForReal << std::endl;

	float* keypoint_data = ReadFeatures(featureFile,

	bagOfWordsFile.open(outputFileForReal);

	for(int i = 0; i < fileNames.size(); ++i)
	{

		//int* nearest =  &nearest_neighbors_result[i];
		//int nearest_num = (int)(*(nearest + 4)); // the cluster center id is stored in the next 4 bytes
		bagOfWordsFile << "<DOC>" << endl;
		bagOfWordsFile << "<DOCNO>" << fileNames[i] << "</DOCNO>" << endl;
		bagOfWordsFile << "<TEXT>" << endl;
		bagOfWordsFile << "beans" << endl;
		bagOfWordsFile << "</TEXT>" << endl;
		bagOfWordsFile << "</DOC>" << endl;

	}

	bagOfWordsFile.close();

	printf("Finished writing file.\n");
}
*/
void ReadSizes(vector<int>* sizes, const char* sizeFile)
{

	std::ifstream sizeFileStream;
	sizeFileStream.open(sizeFile);
	// Each line has a size on it
	if(sizeFileStream.is_open())
	{
		int i = 0;
		while(!sizeFileStream.eof())
		{
			if(i > 100)
				break;
			std::string str;
			sizeFileStream >> str;
			int size = atoi(str.c_str());
			(*sizes).push_back(size);

			++i;
		}
	}else
	{
		std::cout << "Error opening file " << sizeFile << std::endl;
		return;
	}
	std::cout << "Finished reading " << (*sizes).size() << " sizes." << std::endl;
	sizeFileStream.close();
}
float* ReadFeatures(char* featureFile, int totalKeypoints, const int KEYPOINT_DIMENSIONALITY)
{
	float* all_keypoints_flat_data;

	std::ifstream featureFileStream;
	featureFileStream.open(featureFile);
	all_keypoints_flat_data = new float[totalKeypoints * KEYPOINT_DIMENSIONALITY];
	// Each line has a size on it
	if(featureFileStream.is_open())
	{
		printf("Total keypoints to read: %d\n", totalKeypoints);

		int current_keypoint = 0;
		std::string dimension;
		float n;

		while(current_keypoint < totalKeypoints)
		{
			
			for(int i = 0; (i < 50000 && current_keypoint < totalKeypoints); ++i)
			{
				int current_dim = 0;
				int p = current_keypoint * KEYPOINT_DIMENSIONALITY;
				// Read each dimension in the 128 vector
				while(current_dim < KEYPOINT_DIMENSIONALITY / 8)
				{
					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;

					featureFileStream >> n;
					all_keypoints_flat_data[p  + current_dim] = n;
					current_dim++;
				}

				current_keypoint++;
			}
			float percent_complete = (float)current_keypoint / (float)totalKeypoints;
			printf("Percent complete: %f\n",percent_complete);
		}
	}else{
		printf("Error opening file %s\n",featureFile);

	}
	featureFileStream.close();

	printf("Finished reading %d keypoints.\n", totalKeypoints);
	return all_keypoints_flat_data;
}
int ReadClusterFile(char* clusterOutputFile, float* cluster_centers, int* numClusters)
{
	std::ifstream fileStream;
	fileStream.open(clusterOutputFile);
	int length = 0;
	// Each line has a size on it

	if(fileStream.is_open())
	{

		int total_point_dims;
		int num_dimensions;
		fileStream >> total_point_dims;
		fileStream >> num_dimensions;
		length = total_point_dims;

		cluster_centers = new float[length];

		int i = 0;
		std::string str;
		float dim = 0;
		while(i < length)
		{	

			for(int k = 0; k < 100000 && i < length; ++k)
			{
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
	*numClusters = length; // set value of numClusters to length
	std::cout << "Finished reading cluster file. Read " << length << " sizes." << std::endl;
	fileStream.close();
	return 1;
}

EXPORTED void UpdateCluster(char* sizeFile, char* featureFile, char* imgListFile, char* clusterOutputFile, char* bagOfWordsOutputFile)
{

	const int NUM_VISUAL_WORDS = 150000;
	const int KEYPOINT_DIMENSIONALITY = 128;

	const int RESULT_ROWS = NUM_VISUAL_WORDS * KEYPOINT_DIMENSIONALITY;
	float* _centers = new float[0];
	int num_rows;
		
	const char NULL_BYTE[] = { '\0' };
	/*
	strcat(sizeFile, NULL_BYTE);
	strcat(featureFile, NULL_BYTE);
	strcat(imgListFile, NULL_BYTE);
	strcat(clusterOutputFile, NULL_BYTE);
	strcat(bagOfWordsOutputFile, NULL_BYTE);

	strcat(sizeFile, NULL_BYTE);
	strcat(featureFile, NULL_BYTE);
	strcat(imgListFile, NULL_BYTE);
	strcat(clusterOutputFile, NULL_BYTE);
	strcat(bagOfWordsOutputFile, NULL_BYTE);

	strcat(sizeFile, NULL_BYTE);
	strcat(featureFile, NULL_BYTE);
	strcat(imgListFile, NULL_BYTE);
	strcat(clusterOutputFile, NULL_BYTE);
	strcat(bagOfWordsOutputFile, NULL_BYTE);
	*/
	int cr = ReadClusterFile(clusterOutputFile, _centers, &num_rows);
	if(cr == 1)
	{
		printf("Cluster file read successfully.\n");
		IndexParameters build_index_params;
		build_index_params.algorithm = KDTREE;
		build_index_params.checks = 2048;
		build_index_params.trees = 8;
		build_index_params.target_precision = -1;
		build_index_params.build_weight = 0.01;
		build_index_params.memory_weight = 1;

		float speedup;
		printf("Created %d rows.\n",num_rows);
		FLANN_INDEX index = flann_build_index(_centers,10,KEYPOINT_DIMENSIONALITY,&speedup, &build_index_params,NULL);
		printf("Built index.\n");
			const char null_byte[] = { '\0' };
		
		std::ofstream bagOfWordsFile;
		//char* outputFile = bagOfWordsOutputFile;
		char* outputFile = new char[strlen(bagOfWordsOutputFile) + 1];
		
		strcpy(outputFile, bagOfWordsOutputFile);

		std::ifstream imgListFileStream;
		std::string imgListFileStr(imgListFile);
		imgListFileStream.open(imgListFileStr.c_str());

		vector<std::string> fileNames;

		printf("Reading in image file names\n");
		if(imgListFileStream.is_open())
		{
			while(!imgListFileStream.eof())
			{
				std::string str;
				imgListFileStream >> str;
				fileNames.push_back(str);
			}
		}else{
			printf("Error opening file %s\n",imgListFile);
			return;
		}

		std::stringstream strstream;
		strstream << "bagofwords.txt";
		const char* doc_num = strstream.str().c_str();
		std::string beans(outputFile);
		std::string backslash("\\");
		std::string outputFileForReal = outputFile + backslash + strstream.str();
		std::cout << outputFileForReal << std::endl;

		vector<int> sizes; // says how many 128 vectors a document has (for each element)
		std::string sizeFileStr(sizeFile);
		ReadSizes(&sizes, sizeFileStr.c_str());

		int total_keypoints_possible = 0;
		for(int i = 0; i < sizes.size(); ++i)
		{
			total_keypoints_possible += sizes[i];
		}
		float* keypoint_data = ReadFeatures(featureFile, total_keypoints_possible, KEYPOINT_DIMENSIONALITY);

		bagOfWordsFile.open(outputFileForReal);
		
		FLANNParameters flann_params;
		flann_params.log_level = LOG_NONE;
		flann_params.log_destination = NULL;
		flann_params.random_seed = CENTERS_RANDOM;
		int keypoints_examined = 0;
		for(int i = 0; i < fileNames.size(); ++i)
		{

			//int* nearest =  &nearest_neighbors_result[i];
			//int nearest_num = (int)(*(nearest + 4)); // the cluster center id is stored in the next 4 bytes
			bagOfWordsFile << "<DOC>" << endl;
			bagOfWordsFile << "<DOCNO>" << fileNames[i] << "</DOCNO>" << endl;
			bagOfWordsFile << "<TEXT>" << endl;

			int keypoints_i = sizes[i];
			for(int j = 0; j < sizes[i]; ++j)
			{
				float* keypoint = new float[KEYPOINT_DIMENSIONALITY];
				for(int k = 0; k < KEYPOINT_DIMENSIONALITY; ++k)
				{
					keypoint[k] = keypoint_data[keypoints_examined * KEYPOINT_DIMENSIONALITY + k];
				}
				keypoints_examined++;
				int* nearest_neighbor = new int[1];
				int _r = flann_find_nearest_neighbors_index(index,keypoint,1,nearest_neighbor,1,1024,&flann_params);
				bagOfWordsFile << "w" << nearest_neighbor[0] << " ";
			}
			bagOfWordsFile << "</TEXT>" << endl;
			bagOfWordsFile << "</DOC>" << endl;

		}

		bagOfWordsFile.close();

		printf("Finished writing file.\n");

		return;
	}else{
		printf("Failed to read cluster file.\n");
		return;
	}

	vector<int> sizes; // says how many 128 vectors a document has (for each element)

	ReadSizes(&sizes, sizeFile);

	int total_keypoints_possible = 0;
	for(int i = 0; i < sizes.size(); ++i)
	{
		total_keypoints_possible += sizes[i];
	}

	//int total_dims = all_keypoints.size() * KEYPOINT_DIMENSIONALITY;
	float* flann_data = ReadFeatures(featureFile, total_keypoints_possible, KEYPOINT_DIMENSIONALITY); // linear data in row-major order

	float* cluster_centers = new float[RESULT_ROWS]; // RESULT_ROWS of KEYPOINT_DIMENSIONALITY vectors that are the centers of every cluster
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
	float speedup;

	printf("Cols: %d, Rows: %d\n", KEYPOINT_DIMENSIONALITY, total_keypoints_possible);
	int r = flann_compute_cluster_centers(flann_data, total_keypoints_possible, KEYPOINT_DIMENSIONALITY, NUM_VISUAL_WORDS, cluster_centers, &index_params, NULL);
	printf("Cluster centers computed. Return result was %d\n",r);

	std::ofstream clusterCenterFileStream;
	clusterCenterFileStream.open(clusterOutputFile);

	/*
		File Format:
		1st line: How many rows 
		2nd line: how many numbers for each row (vector dimensionality)
		3rd - end: rows of data
	*/
	printf("Writing cluster file\n");
	clusterCenterFileStream << (r * KEYPOINT_DIMENSIONALITY) << endl;
	clusterCenterFileStream << KEYPOINT_DIMENSIONALITY << endl;

	int dim = 0;
	for(int i = 0; i < (r * KEYPOINT_DIMENSIONALITY); ++i)
	{
		clusterCenterFileStream << cluster_centers[i] << " ";
		if(dim >= 128)
		{
			clusterCenterFileStream << endl;
			dim = 0;
		}
		++dim;
	}
	
	clusterCenterFileStream.close();
	//FLANN_INDEX index = flann_build_index(result, row_size, column_size, &speedup, &build_index_params, NULL);
	printf("Complete.\n",r);
	/*
	WriteBagOfWords(imgListFile, bagOfWordsOutputFile);
	
	IndexParameters build_index_params;
	build_index_params.algorithm = KDTREE;
	build_index_params.checks = 2048;
	build_index_params.trees = 8;
	build_index_params.target_precision = -1;
	build_index_params.build_weight = 0.01;
	build_index_params.memory_weight = 1;
	
	FLANNParameters flann_params;
	flann_params.log_level = LOG_NONE;
	flann_params.log_destination = NULL;
	flann_params.random_seed = CENTERS_RANDOM;

	printf("Building index.\n");
	FLANN_INDEX index = flann_build_index(result, RESULT_ROWS, KEYPOINT_DIMENSIONALITY, &speedup, &build_index_params, NULL);

	int* nearest_neighbors_result = new int[row_size];
	printf("Index complete.\n");
	*/

	/*
	printf("Searching for nearest neighbors\n");
	//int r2 = flann_find_nearest_neighbors_index(index, flann_data, total_keypoints_possible, nearest_neighbors_result, 1, 1024, &flann_params);
	printf("Nearest neighbor search complete. Returned code %d\n", r2);
	*/


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
