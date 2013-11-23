using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

using Lucene.Net.Index;
using Lucene.Net.Analysis;
using Lucene.Net.Store;
using Lucene.Net.Documents;
using Lucene.Net.Linq;
using Lucene.Net.QueryParsers;
using Lucene.Net.Search;

using System.Collections;

using KeypointExtraction;
using ImageSearchEngine.Util;

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

        const string INDEX_DIRECTORY = @"C:\Users\Raider\ImageSearchFinalProject\LuceneIndex\Index";
        const string DOCUMENT_DIRECTORY = @"C:\Users\Raider\ImageSearchFinalProject\LuceneIndex\Collection\coll";

        public ActionResult IndexDocuments()
        {
            DateTime start = DateTime.Now;

            Analyzer analyzer = new StopAnalyzer(Lucene.Net.Util.Version.LUCENE_30);
            Directory dir = FSDirectory.Open(INDEX_DIRECTORY);
            Directory docDirectory = FSDirectory.Open(DOCUMENT_DIRECTORY);
            IndexWriter writer = new IndexWriter(dir, analyzer, true, IndexWriter.MaxFieldLength.UNLIMITED);

            string[] files = docDirectory.ListAll();
            foreach (string f in files)
            {
                TrecDocIterator iterator = new TrecDocIterator(DOCUMENT_DIRECTORY + "\\" + f);
                Document d;
                while (iterator.MoveNext())
                {
                    d = iterator.Current;
                    if (d != null && d.GetField("contents") != null)
                    {
                        writer.AddDocument(d);
                    }
                }
            }

            writer.Dispose();

            double finish = (DateTime.Now - start).TotalSeconds;
            //TrecDocIterator iterator = new TrecDocIterator(

            return View(model:"Index took " + finish + " seconds.");
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

            return Json("bingo");
            /*
            string keyFile = KeypointExtractor.ConvertPGMToKeyPoints(queryImage.InputStream, queryId, pathToWinSift, pathToQueryFolder);
            List<Keypoint128> keypoints = KeypointExtractor.GetKeypointsFromKeyFile(keyFile);
            return Json(new ImageQuery()
                {
                    Id = queryId,
                    Keypoints = keypoints
                }
            );*/
        }

    }
}
