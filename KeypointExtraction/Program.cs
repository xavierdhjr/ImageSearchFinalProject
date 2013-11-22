using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Runtime.InteropServices;

namespace KeypointExtraction
{
    class Program
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct IndexParameters
        {
            public int algorithm;
            public int checks;
            public float cb_index;
            public int trees;
            public int branching;
            public int iterations;
            public int centers_init;
            public float target_precision;
            public float build_weight;
            public float memory_weight;
            public float sample_fraction;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct FLANNParameters
        {
            public int log_level;
            public char[] log_destination;
            public long random_seed;
        }



        [DllImport("FLANNDLL.dll", CallingConvention= CallingConvention.Cdecl)]
        public static extern void UpdateCluster(char[] sizeFile, char[] featureFile, char[] imgListFile, char[] clusterOutputFile, char[] bagOfWordsOutputDir);

        [DllImport("FLANNDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void UpdateClusterCenters(char[] sizeFile, char[] featureFile, char[] clusterOutputFile);

        [DllImport("FLANNDLL.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string CreateBagOfWords();
        /*
        [DllImport("FLANNDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int* FindNearestNeighbors(char[] clusterFile, float[] imageQuery);
        */
        static void Main(string[] args)
        {
            /*
            IndexParameters indexParams = new IndexParameters();
            FLANNParameters flannParams = new FLANNParameters();
            float[] beans;
            int rows = 5;
            int cols = 5;
            System.Random rnd = new Random();
            float[][] doop = new float[rows][];
            for (int i = 0; i < rows; ++i)
            {
                doop[i] = new float[cols];
                for (int k = 0; k < cols; ++k)
                {
                    doop[i][k] = rnd.Next(0, rows);
                }
            }
            int n = flann_compute_cluster_centers(doop, 1, 1, 1, out beans, ref indexParams, ref flannParams);*/
            char[] sizeFile = (Constants.PATH_TO_FEATURES + "/" + Constants.SIZE_FILE).ToCharArray();
            char[] featuresFile = (Constants.PATH_TO_FEATURES + "/" + Constants.FEATURES_FILE).ToCharArray();
            char[] imgListFile = (Constants.PATH_TO_FEATURES + "/" + Constants.IMGLIST_FILE).ToCharArray();
            char[] clusterOutputFile = (Constants.PATH_TO_CLUSTERS_FILE.ToCharArray());
            /*
            int dimensionality = 1259;
            float[] query = new float[1259];
            System.Random rng = new Random();
            for (int i = 0; i < dimensionality; ++i)
            {
                query[i] = rng.Next(1259);
            }*/
            /*
            var incoming = new int[1259];

            
            unsafe
            {
                int* result = FindNearestNeighbors(Constants.PATH_TO_CLUSTERS_FILE.ToCharArray(), query);
                for (int i = 0; i < incoming.Length; ++i)
                {
                    incoming[i] = result[i];
                }
            }*/
            //UpdateCluster(sizeFile, featuresFile, imgListFile, Constants.PATH_TO_CLUSTERS_FILE.ToCharArray(), Constants.PATH_TO_BAGOFWORDS.ToCharArray());
            //UpdateClusterCenters(sizeFile, featuresFile, clusterOutputFile);
            string soup = CreateBagOfWords();
        }
    }
}
