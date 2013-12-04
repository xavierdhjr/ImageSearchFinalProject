using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Collections;

using Lucene.Net.Documents;
using Lucene.Net.Store;
using System.IO;
using System.Text.RegularExpressions;
using System.Text;

namespace ImageSearchEngine.Util
{
    public class TrecDocIterator : IEnumerator<Document>
    {
        public Document Current
        {
            get
            {
                Document doc = new Document();
                StreamReader reader = _reader;
                StringBuilder builder = new StringBuilder();
                //StringReader stringReader;
                try
                {
                    string line;
                    Regex pattern = new Regex("<DOCNO>\\s*(\\S+)\\s*<");
                    bool in_doc = false;
                    while (true)
                    {
                        line = reader.ReadLine();
                        if (line == null)
                        {
                            _eof = true;
                            break;
                        }
                        if (!in_doc)
                        {

                            if (line.StartsWith("<DOC>"))
                                in_doc = true;
                            else
                                continue;
                        }
                        if (line.StartsWith("</DOC>"))
                        {
                            in_doc = false;
                            builder.Append(line);
                            break;
                        }

                        Match match = pattern.Match(line);
                        if (match.Success)
                        {
                            string docno = match.Groups[1].Value;
                            doc.Add(new Lucene.Net.Documents.Field("docno",docno,Field.Store.YES, Field.Index.NOT_ANALYZED));
                        }
                        builder.Append(line);
                    }

                    if(builder.Length > 0)
                    {
                        doc.Add(new Lucene.Net.Documents.Field("contents",builder.ToString(),Field.Store.NO, Field.Index.ANALYZED));
                    }
                }
                catch (Exception e)
                {
                    doc = null;
                }
                ++i;
                return doc;
            }
        }

        FileStream _fs;
        StreamReader _reader;
        string _filePath;
        bool _eof;

        int i = 0;
        public int DocsIterated
        {
            get { return i; }
        }
        public TrecDocIterator(string pathToFile)
        {
            _filePath = pathToFile;
            Reset();
        }

        public bool MoveNext()
        {
            return !_eof;
        }

        public void Reset()
        {
            _eof = false;

            _fs = File.OpenRead(_filePath);
            _reader = new StreamReader(_fs);
        }

        public void Dispose()
        {
            _reader.Dispose();
        }

        object IEnumerator.Current
        {
            get { return Current; }
        }
    }
}