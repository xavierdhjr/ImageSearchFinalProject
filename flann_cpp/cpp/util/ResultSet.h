#ifndef RESULTSET_H
#define RESULTSET_H


#include <algorithm>
#include <limits>
#include "../algorithms/dist.h"

using namespace std;


/* This record represents a branch point when finding neighbors in
	the tree.  It contains a record of the minimum distance to the query
	point, as well as the node at which the search resumes.
*/

template <typename T>
struct BranchStruct {
	T node;           /* Tree node at which search resumes */
	float mindistsq;     /* Minimum distance to query for all nodes below. */
	
	bool operator<(const BranchStruct<T>& rhs) 
	{ 
        return mindistsq<rhs.mindistsq;
	}

    static BranchStruct<T> make_branch(T aNode, float dist)
    {
        BranchStruct<T> branch;
        branch.node = aNode;
        branch.mindistsq = dist;
        return branch;
    }
}; 






class ResultSet 
{
	int* indices;
	float* dists;
    int capacity;
	
	float* target;
    int veclen;
	
	int count;
	


public:		
	ResultSet(int capacity_, float* target_ = NULL, int veclen_ = 0 ) : 
        capacity(capacity_), target(target_), veclen(veclen_), count(0)
	{
        indices = new int[capacity_];
        dists = new float[capacity_];            
	}
	
	~ResultSet()
	{
		delete[] indices;
		delete[] dists;
	}

	
	void init(float* target_, int veclen_) 
	{
        target = target_;
        veclen = veclen_;
        count = 0;
	}
	
	
	int* getNeighbors() const
	{	
		return indices;
	}
	
	bool full() const
	{	
		return count == capacity;
	}
	
    
	bool addPoint(float* point, int index) 
	{
		for (int i=0;i<count;++i) {
			if (indices[i]==index) return false;
		}
		float dist = squared_dist(target,point,veclen);
		
		if (count<capacity) {
			indices[count] = index;
			dists[count] = dist;	
			++count;
		} 
		else if (dist < dists[count-1] || (dist == dists[count-1] && index < indices[count-1])) {
//         else if (dist < dists[count-1]) {
			indices[count-1] = index;
			dists[count-1] = dist;
		}
		else { 
			return false;
		}
		
		int i = count-1;
		// bubble up
		while (i>=1 && (dists[i]<dists[i-1] || (dists[i]==dists[i-1] && indices[i]<indices[i-1]) ) ) {
//         while (i>=1 && (dists[i]<dists[i-1]) ) {
			swap(indices[i],indices[i-1]);
			swap(dists[i],dists[i-1]);
			i--;
		}
		
		return true;
	}
	
	float worstDist()
	{
		return (count<capacity) ? numeric_limits<float>::max()  : dists[count-1];
	}
	
};

#endif //RESULTSET_H
