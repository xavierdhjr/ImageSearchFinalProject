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

        private class BM25Similarity : Similarity
        {

            public override float Idf(int docFreq, int numDocs)
            {
                //float retval =  (1 + (numDocs - docFreq + 0.5f)) / docFreq + 1;
                return (float)Math.Log(1 + (numDocs - docFreq + 0.5f) / (docFreq + 0.5f)) + 1;
            }
            public override float SloppyFreq(int distance)
            {
                return 1 / (distance + 1);
            }


            public override float Coord(int overlap, int maxOverlap)
            {
                return (float)Math.Pow((float)overlap / (float)maxOverlap,2);
            }

            public override float LengthNorm(string fieldName, int numTokens)
            {
                return 1;
            }

            public override float QueryNorm(float sumOfSquaredWeights)
            {
                return 1;
            }

            public override float Tf(float freq)
            {
                return freq;
            }
        }
        //
        // GET: /Home/

        const string INDEX_DIRECTORY = @"C:\Users\Raider\Desktop\MSU\FS13\CSE484\project\index";
        const string DOCUMENT_DIRECTORY = @"C:\Users\Raider\Desktop\MSU\FS13\CSE484\project\bagofwords";

        Similarity similarity;

        public HomeController()
        {
            //similarity = new NoNormSimilarity();
            similarity = new DefaultSimilarity();
        }

        public ActionResult Browse(int page = 0,int results = 20)
        {
            IndexReader reader = DirectoryReader.Open(FSDirectory.Open(INDEX_DIRECTORY), true);

            List<Document> documents = new List<Document>();
            List<BrowseImageResult> imagePaths = new List<BrowseImageResult>();

            int offset = page * results;
            int i = 0;
            while (i < results)
            {
                imagePaths.Add
                (
                    new BrowseImageResult()
                    {
                    PathToImage = Url.Content(Constants.VIRTUALPATH_TO_IMAGES + reader.Document(i + offset).Get("docno")),
                    ImageIndexId = i + offset
                    }
                );
                i++;
            }

            BrowseModel model = new BrowseModel()
            {
                page = page,
                resultsPerPage = results,
                Images = imagePaths,
                MaxPages = reader.MaxDoc / results
            };

            return View(model);
        }

        public ActionResult IndexDocuments()
        {
            DateTime start = DateTime.Now;

            Analyzer analyzer = new WhitespaceAnalyzer();// new WhitespaceAnalyzer();// new StopAnalyzer(Lucene.Net.Util.Version.LUCENE_30);
            Directory dir = FSDirectory.Open(INDEX_DIRECTORY);
            Directory docDirectory = FSDirectory.Open(DOCUMENT_DIRECTORY);
            IndexWriter writer = new IndexWriter(dir, analyzer, true, IndexWriter.MaxFieldLength.UNLIMITED);
            writer.SetSimilarity(similarity);
            string[] files = docDirectory.ListAll();
            foreach (string f in files)
            {
                using (TrecDocIterator iterator = new TrecDocIterator(DOCUMENT_DIRECTORY + "\\" + f))
                {

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

        

        public ActionResult FindSimilar(int docid)
        {
            IndexReader reader = DirectoryReader.Open(FSDirectory.Open(INDEX_DIRECTORY), true);

            DateTime start = DateTime.Now;

            IndexSearcher searcher = new IndexSearcher(reader);
            Document document = searcher.Doc(docid);

            Similarity sim = similarity;
            Analyzer analyzer = new WhitespaceAnalyzer();
            searcher.Similarity = sim;

            QueryParser parser = new QueryParser(Lucene.Net.Util.Version.LUCENE_30, "contents", analyzer);
            Directory dir = FSDirectory.Open(INDEX_DIRECTORY);
            Directory docDirectory = FSDirectory.Open(DOCUMENT_DIRECTORY);

            string[] files = docDirectory.ListAll();
            string contents = "";
            foreach (string f in files)
            {
                TrecDocIterator iterator = new TrecDocIterator(DOCUMENT_DIRECTORY + "\\" + f);
                Document d;
                while (iterator.MoveNext())
                {
                    d = iterator.Current;
                    if (d != null && d.GetField("contents") != null)
                    {
                        if (d.Get("docno").Equals(document.Get("docno")))
                        {
                            contents = d.Get("contents");
                            int textIndex = contents.IndexOf("<TEXT>");
                            contents = contents.Substring(textIndex + 6);
                            int endTextIndex = contents.IndexOf("</TEXT>");
                            contents = contents.Substring(0, endTextIndex);
                        }
                    }
                }
            }



            Query q = getQueryFromBagOfWords(contents, analyzer, searcher);

            List<string> similarImages = doBatchSearch(null, searcher, "", q, "");

            List<string> returnedSimilarImages = new List<string>();
            
            for (int i = 0; i < Math.Min(similarImages.Count, 10); ++i)
            {
               

                returnedSimilarImages.Add(Url.Content(Constants.VIRTUALPATH_TO_IMAGES + similarImages[i]));
            }
            TimeSpan time = DateTime.Now - start;

            SimilarImageResult result = new SimilarImageResult()
            {
                PathToQueryImage = Url.Content(Constants.VIRTUALPATH_TO_IMAGES + document.Get("docno")),
                QueryTimeMilliseconds = (int)time.TotalMilliseconds,
                QueryTimeSeconds = (int)time.TotalSeconds,
                SimilarImages = returnedSimilarImages

            };
            return View(result);
        }

        private Query getQueryFromBagOfWords(string bagOfWords, Analyzer analyzer, IndexSearcher searcher)
        {
            System.IO.StringReader strReader = new System.IO.StringReader(bagOfWords);
            QueryParser parser = new QueryParser(Lucene.Net.Util.Version.LUCENE_30, "contents", analyzer);

            Query query = parser.Parse(bagOfWords);

            return query;
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
            Similarity sim = similarity;
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

            return View("FindSimilar",result);
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

        public class BrowseImageResult
        {
            public string PathToImage;
            public int ImageIndexId;
        }

        public class BrowseModel
        {
            public List<BrowseImageResult> Images;
            public int page;
            public int resultsPerPage;
            public int MaxPages;
        }

    }
}
