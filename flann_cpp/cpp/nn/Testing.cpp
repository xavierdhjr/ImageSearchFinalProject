#include "Testing.h"

#include "../util/ResultSet.h"
#include "../util/Timer.h"
#include "../util/Logger.h"
#include "../algorithms/dist.h"
#include "../util/common.h"

#include <algorithm>
#include <math.h>



const float SEARCH_EPS = 0.001;

int countCorrectMatches(int* neighbors, int* groundTruth, int n)
{
    int count = 0;
    for (int i=0;i<n;++i) {
        for (int k=0;k<n;++k) {
            if (neighbors[i]==groundTruth[k]) {
                count++;
                break;
            }
        }
    }
    return count;
}


float computeDistanceRaport(const Dataset<float>& inputData, float* target, int* neighbors, int* groundTruth, int veclen, int n)
{
    float ret = 0;
    for (int i=0;i<n;++i) {
        float den = squared_dist(target,inputData[groundTruth[i]], veclen);
        float num = squared_dist(target,inputData[neighbors[i]], veclen);

//        printf("den=%g,num=%g\n",den,num);
        
        if (den==0 && num==0) {
            ret += 1;
        } else {
            ret += num/den;
        }
    }
    
    return ret;
}

float search_with_ground_truth(NNIndex& index, const Dataset<float>& inputData, const Dataset<float>& testData, const Dataset<int>& matches, int nn, int checks, float& time, float& dist, int skipMatches) 
{
    if (matches.cols<nn) {
        logger.info("matches.cols=%d, nn=%d\n",matches.cols,nn);
        
        throw FLANNException("Ground truth is not computed for as many neighbors as requested");
    }
    
    ResultSet resultSet(nn+skipMatches);
    Params searchParams;
    searchParams["checks"] = checks;

    int correct;
    float distR;
    StartStopTimer t;
    int repeats = 0;
    while (t.value<0.2) {
        repeats++;
        t.start();
        correct = 0;
        distR = 0;
        for (int i = 0; i < testData.rows; i++) {
            float* target = testData[i];
            resultSet.init(target, testData.cols);
            index.findNeighbors(resultSet,target, searchParams);            
            int* neighbors = resultSet.getNeighbors();
            neighbors = neighbors+skipMatches;
                    
            correct += countCorrectMatches(neighbors,matches[i], nn);
            distR += computeDistanceRaport(inputData, target,neighbors,matches[i], testData.cols, nn);
        }
        t.stop();
    }
    time = t.value/repeats;
    
    
    float precicion = (float)correct/(nn*testData.rows);

    dist = distR/(testData.rows*nn);
    
    logger.info("%8d %10.4g %10.5g %10.5g %10.5g\n",
            checks, precicion, time, 1000.0 * time / testData.rows, dist);
    
    return precicion;
}


void search_for_neighbors(NNIndex& index, const Dataset<float>& testset, Dataset<int>& result, Params searchParams, int skip)
{
    assert(testset.rows == result.rows);

    int nn = result.cols;
    ResultSet resultSet(nn+skip);

    for (int i = 0; i < testset.rows; i++) {
        float* target = testset[i];
		//printf("Target found [%d]\n",i);
        resultSet.init(target, testset.cols);
                
        index.findNeighbors(resultSet,target, searchParams);
        
        int* neighbors = resultSet.getNeighbors();
        memcpy(result[i], neighbors+skip, nn*sizeof(int));        
		//printf("Neighbor found: %d\n",*neighbors);
    }

}

float test_index_checks(NNIndex& index, const Dataset<float>& inputData, const Dataset<float>& testData, const Dataset<int>& matches, int checks, float& precision, int nn, int skipMatches)
{
    logger.info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist\n");
    logger.info("---------------------------------------------------------\n");
    
    float time = 0;
    float dist = 0;
    precision = search_with_ground_truth(index, inputData, testData, matches, nn, checks, time, dist, skipMatches);

    return time;
}


float test_index_precision(NNIndex& index, const Dataset<float>& inputData, const Dataset<float>& testData, const Dataset<int>& matches,
             float precision, int& checks, int nn, int skipMatches)
{       
    logger.info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist\n");
    logger.info("---------------------------------------------------------\n");
    
    int c2 = 1;
    float p2;
    int c1;
    float p1;
    float time;
    float dist;
    
    p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);
    
    if (p2>precision) {
        logger.info("Got as close as I can\n");
        checks = c2;
        return time;
    }

    while (p2<precision) {
        c1 = c2;
        p1 = p2;
        c2 *=2;
        p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);
    }   
    
    int cx;
    float realPrecision;
    if (fabs(p2-precision)>SEARCH_EPS) {
        logger.info("Start linear estimation\n");
        // after we got to values in the vecinity of the desired precision
        // use linear approximation get a better estimation
            
        cx = (c1+c2)/2;
        realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
        while (fabs(realPrecision-precision)>SEARCH_EPS) {
            
            if (realPrecision<precision) {
                c1 = cx;
            }
            else {
                c2 = cx;
            }
            cx = (c1+c2)/2;
            if (cx==c1) {
                logger.info("Got as close as I can\n");
                break;
            }
            realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
        }
        
        c2 = cx;
        p2 = realPrecision;
        
    } else {
        logger.info("No need for linear estimation\n");
        cx = c2;
        realPrecision = p2;
    }
        
    checks = cx;
    return time;
}


float test_index_precisions(NNIndex& index, const Dataset<float>& inputData, const Dataset<float>& testData, const Dataset<int>& matches,
                    float* precisions, int precisions_length, int nn, int skipMatches, float maxTime)
{   
    // make sure precisions array is sorted 
    sort(precisions, precisions+precisions_length);

    int pindex = 0;
    float precision = precisions[pindex];
    
    logger.info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist");
    logger.info("---------------------------------------------------------");
    
    int c2 = 1;
    float p2;
    
    int c1;
    float p1;
    
    float time;
    float dist;
    
    p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);
    
    // if precision for 1 run down the tree is already
    // better then some of the requested precisions, then
    // skip those
    while (precisions[pindex]<p2 && pindex<precisions_length) {
        pindex++;
    }
    
    if (pindex==precisions_length) {
        logger.info("Got as close as I can\n");
        return time;
    }
    
    for (int i=pindex;i<precisions_length;++i) {
    
        precision = precisions[i];
        while (p2<precision) {
            c1 = c2;
            p1 = p2;
            c2 *=2;
            p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);
            if (maxTime> 0 && time > maxTime && p2<precision) return time;
        }
        
        int cx;
        float realPrecision;
        if (fabs(p2-precision)>SEARCH_EPS) {
            logger.info("Start linear estimation\n");
            // after we got to values in the vecinity of the desired precision
            // use linear approximation get a better estimation
                
            cx = (c1+c2)/2;
            realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
            while (fabs(realPrecision-precision)>SEARCH_EPS) {
                
                if (realPrecision<precision) {
                    c1 = cx;
                }
                else {
                    c2 = cx;
                }
                cx = (c1+c2)/2;
                if (cx==c1) {
                    logger.info("Got as close as I can\n");
                    break;
                }
                realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
            }
            
            c2 = cx;
            p2 = realPrecision;
            
        } else {
            logger.info("No need for linear estimation\n");
            cx = c2;
            realPrecision = p2;
        }
        
    }
    return time;
}
