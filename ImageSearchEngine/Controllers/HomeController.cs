using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

using KeypointExtraction;

namespace ImageSearchEngine.Controllers
{
    public class HomeController : Controller
    {
        //
        // GET: /Home/

        public class ImageQuery
        {
            public Guid Id;
            public List<Keypoint128> Keypoints;

        }

        public ActionResult Index()
        {
            return View();
        }

        public JsonResult Query(HttpPostedFileBase queryImage)
        {
            Guid queryId = Guid.NewGuid();
            string winSiftFileName = "~/siftWin32.exe";
            string pathToWinSift = HttpContext.Server.MapPath(winSiftFileName);
            string queryFolder = "~/queries/";
            string pathToQueryFolder = HttpContext.Server.MapPath(queryFolder);

            string keyFile = KeypointExtractor.ConvertPGMToKeyPoints(queryImage.InputStream, queryId, pathToWinSift, pathToQueryFolder);
            List<Keypoint128> keypoints = KeypointExtractor.GetKeypointsFromKeyFile(keyFile);
            return Json(new ImageQuery()
                {
                    Id = queryId,
                    Keypoints = keypoints
                }
            );
        }

    }
}
