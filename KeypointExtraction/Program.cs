using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Runtime.InteropServices;
using System.IO;

using System.Drawing;
using System.Drawing.Drawing2D;

using Adrian.Imaging.PGMConverter;
using OpenSURFcs;

using System.Runtime.InteropServices;

using OpenCV.Net;

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

        [DllImport("FLANNDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string CreateBagOfWords(float[] keypoint_data, int num_keypoints);

        [DllImport("FLANNDLL.dll", CallingConvention= CallingConvention.Cdecl)]
        public static extern void UpdateCluster(char[] sizeFile, char[] featureFile, char[] imgListFile, char[] clusterOutputFile, char[] bagOfWordsOutputDir);

        [DllImport("FLANNDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void UpdateClusterCenters(char[] sizeFile, char[] featureFile, char[] clusterOutputFile);
        /*
        [DllImport("FLANNDLL.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string CreateBagOfWords();*/
        /*
        [DllImport("FLANNDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int* FindNearestNeighbors(char[] clusterFile, float[] imageQuery);
        */
        public static void doUpdateclusters()
        {
            char[] sizeFile = (Constants.PATH_TO_FEATURES + "/" + Constants.SIZE_FILE).ToCharArray();
            char[] featuresFile = (Constants.PATH_TO_FEATURES + "/" + Constants.FEATURES_FILE).ToCharArray();
            char[] imgListFile = (Constants.PATH_TO_FEATURES + "/" + Constants.IMGLIST_FILE).ToCharArray();
            char[] clusterOutputFile = (Constants.PATH_TO_CLUSTERS_FILE.ToCharArray());
            testImage();
            //UpdateClusterCenters(sizeFile, featuresFile, clusterOutputFile);
        }

        public static void testImage()
        {
            List<string> similarImages = new List<string>();
            string pgmFileName = @"C:\Users\Raider\Documents\Visual Studio 2012\Projects\ImageSearchEngine\ImageSearchEngine\queries\5db64103-80f1-4787-96ae-faa730dc87b9.pgm";
            Bitmap pgmConvertedToBitmap = PGMUtil.ToBitmap(pgmFileName);

            IntegralImage iimg = IntegralImage.FromImage(pgmConvertedToBitmap);
            List<IPoint> ipts = FastHessian.getIpoints(0.0002f, 5, 2, iimg);
            Image tempImg = new Bitmap(pgmConvertedToBitmap);

            SurfDescriptor.DecribeInterestPoints(ipts, false, true, iimg); // 128 length descriptor

            int keypointSize = 0;
            for (int i = 0; i < ipts.Count; ++i)
            {
                keypointSize += ipts[i].descriptorLength;

            }

            float[] keypoint_data = new float[keypointSize];
            int k = 0;
            for (int i = 0; i < ipts.Count; ++i)
            {
                for (int j = 0; j < ipts[i].descriptorLength; ++j)
                {
                    keypoint_data[k] = ipts[i].descriptor[j];
                    ++k;
                }
            }

            string result = CreateBagOfWords(keypoint_data, keypointSize);

        }

        static void Main(string[] args)
        {
            doUpdateclusters();
        }
    }
}
