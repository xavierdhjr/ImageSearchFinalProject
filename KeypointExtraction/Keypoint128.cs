using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KeypointExtraction
{
    /// <summary>
    /// Represents a keypoint with a 128 descriptor vector
    /// </summary>
    public class Keypoint128
    {
        public byte[] descriptor;
        public float subpixelRow;
        public float subpixelColumn;
        public float scale;
        public float orientationInRadians;

        public Keypoint128()
        {
            descriptor = new byte[128];
        }

    }
}
