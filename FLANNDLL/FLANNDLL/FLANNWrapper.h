#ifndef FLANNWRAPPER_H
#define FLANNWRAPPER_H


// lol so jank
#include "..\..\..\..\..\..\Downloads\cse484project\flann\source code\flann-1.1-win32\src\cpp\flann.h"
#include <string>

#ifdef WIN32
/* win32 dll export/import directives */
#ifdef flann_EXPORTS
#define LIBSPEC __declspec(dllexport)
#else
#define LIBSPEC __declspec(dllimport)
#endif
#else
/* unix needs nothing */
#define LIBSPEC
#endif

#ifdef __cplusplus
extern "C" 
{
#endif

#ifdef __cplusplus
}
#endif


#endif /*FLANNWRAPPER_H*/
