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

        const string INDEX_DIRECTORY = @"C:\Users\Raider\Desktop\MSU\FS13\CSE484\project\index";
        const string DOCUMENT_DIRECTORY = @"C:\Users\Raider\Desktop\MSU\FS13\CSE484\project\bagofwords";

        public ActionResult IndexDocuments()
        {
            DateTime start = DateTime.Now;

            Analyzer analyzer = new WhitespaceAnalyzer();// new StopAnalyzer(Lucene.Net.Util.Version.LUCENE_30);
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

        private List<string> doBatchSearch(System.IO.StringReader strReader, IndexSearcher searcher, string qid, Query query, string runTag)
        {
            TopDocs results = searcher.Search(query, 2000);
            ScoreDoc[] hits = results.ScoreDocs;
            int numTotalHits = results.TotalHits;
            int start = 0;
            int end = Math.Min(numTotalHits, 1000);

            Dictionary<string, string> seen = new Dictionary<string, string>();
            List<string> seenList = new List<string>();
            for (int i = start; i < end; ++i)
            {
                Document doc = searcher.Doc(hits[i].Doc);
                string docno = doc.Get("docno");
                if (seen.ContainsKey(docno))
                    continue;
                seen.Add(docno, docno);
                seenList.Add(docno);

            }
            return seenList;
        }

        public ActionResult Query(HttpPostedFileBase queryImage)
        {
            DateTime start = DateTime.Now;

            Guid queryId = Guid.NewGuid();
            string winSiftFileName = "~/siftWin32.exe";
            string pathToWinSift = HttpContext.Server.MapPath(winSiftFileName);
            string queryFolder = "~/queries/";
            string pathToQueryFolder = HttpContext.Server.MapPath(queryFolder);

            string bagOfWords = KeypointExtractor.GetBagOfWords(queryImage.InputStream, queryId, pathToQueryFolder);
            Similarity sim = new DefaultSimilarity();
            IndexReader reader = DirectoryReader.Open(FSDirectory.Open(INDEX_DIRECTORY),true);
            Analyzer analyzer = new WhitespaceAnalyzer();// new StopAnalyzer(Lucene.Net.Util.Version.LUCENE_30);
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.Similarity = sim;

            System.IO.StringReader strReader = new System.IO.StringReader(bagOfWords);
            QueryParser parser = new QueryParser(Lucene.Net.Util.Version.LUCENE_30, "contents", analyzer);
            int num_queries = 0;
            List<string> results = new List<string>();
            while (true)
            {
                string line = strReader.ReadLine();
                if (line == null || line.Length == -1)
                    break;
                string query_id = null;
                if (line.Equals("<DOC>"))
                {
                    line = strReader.ReadLine();
                    query_id = line.Replace("<DOCNO>", "");
                    query_id = query_id.Replace("</DOCNO>", "");
                    line = strReader.ReadLine();
                    line = strReader.ReadLine();
                }
                else
                {
                    continue;
                }

                line = line.Trim();
                if (line.Length == 0)
                    break;

                line = line.Replace("/", " ");

                Query query = parser.Parse(line);
                num_queries++;
                results = doBatchSearch(strReader, searcher, query_id, query, "default");

                line = strReader.ReadLine();
                line = strReader.ReadLine();
                if (!line.Equals("</DOC>")) break;
            }

            reader.Dispose();
            SimilarImageResult result = new SimilarImageResult();
            result.SimilarImages = new List<string>();
            result.PathToQueryImage = Url.Content(queryFolder + queryId.ToString() + ".png");
            int numImages = 0;
            int imageCap = 10;
            foreach (string str in results)
            {
                if (numImages >= imageCap) 
                    break;

                result.SimilarImages.Add(Url.Content(Constants.VIRTUALPATH_TO_IMAGES + str));
                numImages++;
            }
            TimeSpan timeItTook = DateTime.Now - start;

            result.QueryTimeSeconds = (int)timeItTook.TotalSeconds;
            result.QueryTimeMilliseconds = (int)timeItTook.TotalMilliseconds;

            return View(result);
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

        public class SimilarImageResult
        {
            public string PathToQueryImage;
            public List<string> SimilarImages;
            public int QueryTimeSeconds;
            public int QueryTimeMilliseconds;
        }

    }
}
