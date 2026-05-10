using Microsoft.ML.Data;

namespace Shared.Models
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }

        public float[] Score { get; set; }
    }
}
