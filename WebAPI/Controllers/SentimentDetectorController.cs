using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using Shared.Models;
using WebAPI.Models;

namespace WebAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SentimentDetectorController : ControllerBase
    {
        private readonly PredictionEnginePool<SentimentData, SentimentPrediction> _predictionEnginePool;

        public SentimentDetectorController(
            PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [HttpPost("predict")]
        public ActionResult<SentimentPrediction> PredictSentiment(
            [FromBody] SentimentRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Text))
                return BadRequest("Input text cannot be empty.");

            var input = new SentimentData
            {
                Text = request.Text
            };

            var prediction = _predictionEnginePool.Predict(
                modelName: "sentiment_model",
                example: input
            );

            return Ok(prediction);
        }
    }
}