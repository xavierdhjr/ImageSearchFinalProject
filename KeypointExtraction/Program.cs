using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Runtime.InteropServices;
using System.Diagnostics;

using System.IO;

using System.Drawing;
using System.Drawing.Drawing2D;

using Adrian.Imaging.PGMConverter;
using OpenSURFcs;


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

        public class KeyFileWriter : IDisposable
        {
            StreamWriter _writer;

            public KeyFileWriter(string pathToKeyFile)
            {
                _writer = new StreamWriter(pathToKeyFile, false);
            }

            public void ReceiveData(object sender, DataReceivedEventArgs e)
            {
                _writer.WriteLine(e.Data);
            }

            public void Dispose()
            {
                _writer.Dispose();
            }
        }

        public static void doUpdateclusters()
        {
            char[] sizeFile = (Constants.PATH_TO_FEATURES + "/" + Constants.SIZE_FILE).ToCharArray();
            char[] featuresFile = (Constants.PATH_TO_FEATURES + "/" + Constants.FEATURES_FILE).ToCharArray();
            char[] imgListFile = (Constants.PATH_TO_FEATURES + "/" + Constants.IMGLIST_FILE).ToCharArray();
            char[] clusterOutputFile = (Constants.PATH_TO_CLUSTERS_FILE.ToCharArray());
            testImage();
            testImage();
            testImage();
            //UpdateClusterCenters(sizeFile, featuresFile, clusterOutputFile);
        }

        public static void testImage()
        {
            List<string> similarImages = new List<string>();
            string pgmFileName = @"C:\Users\Raider\Documents\Visual Studio 2012\Projects\ImageSearchEngine\ImageSearchEngine\queries\c1928963-849b-40a8-92cd-a8fc05820d31.pgm";
            string keyFileName = @"C:\Users\Raider\Documents\Visual Studio 2012\Projects\ImageSearchEngine\ImageSearchEngine\queries\c1928963-849b-40a8-92cd-a8fc05820d31.pgm.key";

            string pathToSift = @"C:\Users\Raider\Documents\Visual Studio 2012\Projects\ImageSearchEngine\ImageSearchEngine\siftWin32.exe";

            ProcessStartInfo siftInfo = new ProcessStartInfo()
            {
                FileName = pathToSift,
                RedirectStandardOutput = true,
                RedirectStandardInput = true,
                UseShellExecute = false
            };
            Console.WriteLine("Starting SIFT: ");
            Process sift = Process.Start(siftInfo);
            sift.EnableRaisingEvents = true;

            using (StreamReader readPgmFile = new StreamReader(pgmFileName))
            {
                sift.StandardInput.WriteLine(readPgmFile.ReadToEnd());
                sift.StandardInput.Flush();
                sift.StandardInput.Close();
            }

            using (KeyFileWriter writer = new KeyFileWriter(keyFileName))
            {
                sift.OutputDataReceived += writer.ReceiveData;
                sift.BeginOutputReadLine();
                sift.WaitForExit();
            }
            //string result = sift.StandardOutput.ReadToEnd();
            sift.Close();
            /*
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
            */
            float[] keypoint_data = KeypointExtractor.ReadKeyFile(keyFileName);

            string result = CreateBagOfWords(keypoint_data, keypoint_data.Length);

        }
        /*
        static void sift_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {
            Console.WriteLine("received: " + e.Data);
        }*/

        static void Main(string[] args)
        {
            doUpdateclusters();
        }
    }
}
