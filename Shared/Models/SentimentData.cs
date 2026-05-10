using Microsoft.ML.Data;

namespace Shared.Models
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
