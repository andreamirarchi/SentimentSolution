using Microsoft.Extensions.ML;
using Shared.Models;
using System.Runtime.CompilerServices;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

builder.Services.AddEndpointsApiExplorer();

builder.Services.AddSwaggerGen();

builder.Services
    .AddPredictionEnginePool
    <SentimentData, SentimentPrediction>()
    .FromFile(
        modelName: "sentiment_model",
        filePath: Path.Combine(
            AppContext.BaseDirectory,
            "MLModels",
            "sentiment_model.zip"
        ),
        watchForChanges: true
    );

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();

    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();