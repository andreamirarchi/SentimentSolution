using Microsoft.ML;
using Shared.Models;
using System;
using System.IO;

var mlContext = new MLContext();

//Load data
var dataPath = Path.Combine(AppContext.BaseDirectory, "data", "reviews.csv");
if (!File.Exists(dataPath))
    throw new FileNotFoundException($"File non trovato: {dataPath}");

var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
    path: dataPath,
    hasHeader: true,
    separatorChar: ','
);

//Build Pipeline
var pipeline =
    mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: "Label",
        inputColumnName: nameof(SentimentData.Label) //output column
    )
    .Append(
        mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentData.Text)
        )
    )
    .Append(
        mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy() //algo
    )
    .Append(
        mlContext.Transforms.Conversion.MapKeyToValue(
            outputColumnName: "PredictedLabel" // predicted column
        )
    );

// Train the Model
var mlModel = pipeline.Fit(dataView);

var predictor =
    mlContext.Model.CreatePredictionEngine
    <SentimentData, SentimentPrediction>(mlModel);

var testReview = new SentimentData
{
    //Text = "This product is okay"
    //Text = "Good product and great quality"
    //Text = "Bad product and poor quality"
    Text = "I regret buying this, it broke in 2 hours and support ignored me"
};

var prediction = predictor.Predict(testReview);

Console.WriteLine($"Prediction: {prediction.PredictedLabel}");

Directory.CreateDirectory("MLModels");

var modelPath = Path.Combine(
    "MLModels",
    "sentiment_model.zip"
);

mlContext.Model.Save(
    mlModel,
    dataView.Schema,
    modelPath
);

Console.WriteLine($"Model saved to: {modelPath}");

var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

mlModel = pipeline.Fit(split.TrainSet);

var predictions = mlModel.Transform(split.TestSet);

var metrics = mlContext.MulticlassClassification.Evaluate(
    data: predictions,
    labelColumnName: "Label",
    predictedLabelColumnName: "PredictedLabel"
);

Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:0.###}");
Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:0.###}");