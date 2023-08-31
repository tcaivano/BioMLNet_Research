using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.DataOperationsCatalog;

namespace BioMLNet_Research.Model
{
    internal class ModelTrainer
    {
        public static void TrainModel(string modelLocation, float testFraction)
        {
            MLContext mlContext = new MLContext
            {
                GpuDeviceId = 0,
                FallbackToCpu = false
            };
            IEnumerable<ImageData> images = LoadImagesFromDirectory(modelLocation);
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey")
            .Append(mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: modelLocation,
                inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: testFraction);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            //TODO - test on easy, med, hard datasets and get metrics here
        }

        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
            foreach (var file in files)
            {
                var label = Directory.GetParent(file).Name;
                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }

        private class ImageData
        {
            public string ImagePath { get; set; }
            public string Label { get; set; }
        }

        private class ModelInput
        {
            public byte[] Image { get; set; }
            public UInt32 LabelAsKey { get; set; }
            public string ImagePath { get; set; }
            public string Label { get; set; }
        }

        private class ModelOutput
        {
            public string ImagePath { get; set; }
            public string Label { get; set; }
            public string PredictedLabel { get; set; }
        }
    }
}
