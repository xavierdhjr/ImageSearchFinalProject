using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Web;

using System.Drawing;
using System.Drawing.Drawing2D;

using Adrian.Imaging.PGMConverter;
using OpenSURFcs;

using System.Runtime.InteropServices;

using OpenCV.Net;

namespace KeypointExtraction
{
    public class KeypointExtractor
    {
        
        /// <summary>
        /// Saves a PGM file to the disk, and sends that to the sift tool.
        /// Returns the path to a file with keypoints.
        /// </summary>
        /// <param name="pgmFileBytes"></param>
        /// <returns></returns>
        public static List<IPoint> ConvertPGMToKeyPoints(Stream inputStream, Guid queryId, string winSiftPath, string saveQueryPath)
        {
            
            string pgmFileName = saveQueryPath + queryId + ".pgm";
            string keyFile = pgmFileName + ".key";

            using (FileStream file = new FileStream(pgmFileName, FileMode.Create, FileAccess.ReadWrite))
            {
                inputStream.CopyTo(file);
            }

            Bitmap pgmConvertedToBitmap = PGMUtil.ToBitmap(pgmFileName);
            IntegralImage iimg = IntegralImage.FromImage(pgmConvertedToBitmap);
            List<IPoint> ipts = FastHessian.getIpoints(0.0002f, 5, 1, iimg);
            Image tempImg = new Bitmap(pgmConvertedToBitmap);

            SurfDescriptor.DecribeInterestPoints(ipts, true, true, iimg); // 128 length descriptor

            /*
             * The file format starts with 2 integers giving the total number of
             *  keypoints and the length of the descriptor vector for each keypoint
             *  (128). Then the location of each keypoint in the image is specified by
             *  4 floating point numbers giving subpixel row and column location,
             *  scale, and orientation (in radians from -PI to PI). Finally, the invariant descriptor vector for the keypoint is given as a list of 128 integers in range [0,255]. 
             */

            int keypoints = ipts.Count;
            int descriptorLength = 128;

            StringBuilder keyFileContents = new StringBuilder();

            keyFileContents.AppendFormat("{0} {1}\n", keypoints, descriptorLength);

            foreach (IPoint ipt in ipts)
            {
                keyFileContents.AppendFormat("{0} {1} {2} {3}\n", ipt.x, ipt.y, ipt.scale, ipt.orientation);
                foreach (float f in ipt.descriptor)
                {
                    keyFileContents.Append(" " + f.ToString());
                }
                keyFileContents.Append("\n");
            }

            /*
            ProcessStartInfo info = new ProcessStartInfo()
            {
                CreateNoWindow = true,
                FileName = winSiftPath,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = false,
                UseShellExecute = false,
                WorkingDirectory = saveQueryPath
            };


            Process p = Process.Start(info);

            string fileContent = File.ReadAllText(fileName);
            p.StandardInput.Write(fileContent);
             * */
            using (FileStream keyFileOutput = new FileStream(keyFile, FileMode.Create, FileAccess.ReadWrite))
            {
                byte[] bytes = UTF8Encoding.ASCII.GetBytes(keyFileContents.ToString());
                keyFileOutput.Write(bytes,0,bytes.Length);
                //p.StandardOutput.BaseStream.CopyTo(keyFileOutput);
            }
            /*
            p.WaitForExit(5000);
            p.Close();
            */
            return ipts;
        }
        /*
        public static List<Keypoint128> GetKeypointsFromKeyFile(string keyFile)
        {

            string[] keyFileContents = File.ReadAllLines(keyFile);
            List<Keypoint128> keypoints = new List<Keypoint128>();


            *
             * The file format starts with 2 integers giving the total number of
             *  keypoints and the length of the descriptor vector for each keypoint
             *  (128). Then the location of each keypoint in the image is specified by
             *  4 floating point numbers giving subpixel row and column location,
             *  scale, and orientation (in radians from -PI to PI). Finally, the invariant descriptor vector for the keypoint is given as a list of 128 integers in range [0,255]. 
             *

            for (int i = 0; i < keyFileContents.Length; ++i)
            {

            }

            return keypoints;
        }
    */

        [DllImport("FLANNDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern string CreateBagOfWords(float[] keypoint_data, int num_keypoints);

        public static string GetBagOfWords(Stream inputStream, Guid queryId, string saveQueryPath)
        {

            
            List<string> similarImages = new List<string>();
            string pgmFileName = saveQueryPath + queryId + ".pgm";
            string keyFileName = pgmFileName + ".key";
            string pngFileName = saveQueryPath + queryId + ".png";

            using (FileStream file = new FileStream(pgmFileName, FileMode.Create, FileAccess.ReadWrite))
            {
                inputStream.CopyTo(file);
            }
            
            Bitmap pgmConvertedToBitmap = PGMUtil.ToBitmap(pgmFileName);
            using (FileStream fs = File.Open(pngFileName, FileMode.OpenOrCreate))
            {
                pgmConvertedToBitmap.Save(fs, System.Drawing.Imaging.ImageFormat.Png);
            }


            string pathToSift = HttpContext.Current.Server.MapPath(@"~\siftWin32.exe");

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

            using (Program.KeyFileWriter writer = new Program.KeyFileWriter(keyFileName))
            {
                sift.OutputDataReceived += writer.ReceiveData;
                sift.BeginOutputReadLine();
                sift.WaitForExit();
            }
            //string result = sift.StandardOutput.ReadToEnd();
            sift.Close();

            float[] keypoints = ReadKeyFile(keyFileName);

            string result = CreateBagOfWords(keypoints, keypoints.Length);

            return result;
        }

        public static float[] ReadKeyFile(string keyFilePath)
        {
            float[] keypoint_data = new float[0];
            using (FileStream fs = File.Open(keyFilePath, FileMode.Open, FileAccess.Read))
            {
                StreamReader reader = new StreamReader(fs);
                string line = reader.ReadLine();
                string[] keypointsAndDimensionality = line.Split(' ');
                int keypoints = int.Parse(keypointsAndDimensionality[0]);
                int dimensionality = int.Parse(keypointsAndDimensionality[1]);

                keypoint_data = new float[keypoints * dimensionality];
                int keypointSize = keypoints * dimensionality;

                int keypoints_left = keypoints;
                int k = 0;
                while (keypoints_left > 0)
                {
                    string blah = reader.ReadLine(); // orientation, scale, etc.
                    int dim = dimensionality;
                    while (dim > 0)
                    {
                        string important = reader.ReadLine();
                        string[] pieceOfVector = important.Split(' ');
                        for (int i = 0; i < pieceOfVector.Length; ++i)
                        {
                            if (pieceOfVector[i].Length <= 0) continue;
                            if (pieceOfVector[i] == " ") continue;
                            keypoint_data[k] = float.Parse(pieceOfVector[i]);
                            dim--;
                            k++;
                        }
                    }

                    keypoints_left--;
                }
            }

            return keypoint_data;
        }

    }
}
