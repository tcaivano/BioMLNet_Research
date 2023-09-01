using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.Formats.Asn1;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Keras.Engine;
using static Microsoft.ML.DataOperationsCatalog;
using static TorchSharp.torch.nn;

namespace BioMLNet_Research.Model
{
    internal class ModelTrainer
    {
        public static string TrainModel(string modelLocation, float testFraction, float learningRate, int batchSize, string saveLocation)
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
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                MetricsCallback = (metrics) => Console.Write("\r" + metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            Console.WriteLine("\nTraining....");
            ITransformer trainedModel = trainingPipeline.Fit(trainSet);
            return SaveModel(saveLocation, "", mlContext, trainSet, DateTime.UtcNow, trainedModel);
        }

        public static void EvaluateModelIdentification(string modelLocation, string dataSet)
        {
            MLContext mlContext = new MLContext
            {
                GpuDeviceId = 0,
                FallbackToCpu = false
            };

            //Define DataViewSchema for data preparation pipeline and trained model
            DataViewSchema modelSchema;

            // Load trained model
            ITransformer trainedModel = mlContext.Model.Load(modelLocation, out modelSchema);
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            int correct = 0, incorrect = 0, i = 0;
            IEnumerable<ImageData> images = LoadImagesFromDirectory(dataSet);
            int total = images.Count();
            foreach(var image in images)
            {
                var input = new ModelInput()
                {
                    Label = image.Label,
                    LabelAsKey = UInt32.Parse(image.Label.Remove(0, 7)),
                    Image = System.IO.File.ReadAllBytes(image.ImagePath),
                    ImagePath = image.ImagePath
                };
                var p = predictionEngine.Predict(input);
                if (image.Label == p.PredictedLabel)
                {
                    correct++;
                }
                else
                {
                    incorrect++;
                }
                i++;
                Console.Write($"\rImage {i} of {total}: Correct {correct}, Incorrect {incorrect}, ");
            }

            Console.WriteLine($"\nCorrect {correct}, Incorrect {incorrect}");
            float accuracy = (float)correct/((float)(incorrect+correct));
            Console.WriteLine($"Accuracy Score: {accuracy}");
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

        private class ImagePrediction : ImageData
        {
            public float[]? Score;

            public string? PredictedLabelValue;
        }

        class ModelInput
        {
            public byte[] Image { get; set; }

            public UInt32 LabelAsKey { get; set; }

            public string ImagePath { get; set; }

            public string Label { get; set; }
        }

        class ModelOutput
        {
            public string ImagePath { get; set; }

            public string Label { get; set; }

            public string PredictedLabel { get; set; }
        }


        /// <summary>
        /// Saves the trained model to a file.
        /// </summary>
        /// <param name="backupFileLocation">The backup file location to save the trained model.</param>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="mlContext">The MLContext.</param>
        /// <param name="trainingdata">The training data.</param>
        /// <param name="now">The start time of the training.</param>
        /// <param name="trainedModel">The trained model.</param>
        private static string SaveModel(string backupFileLocation, string modelName, MLContext mlContext, IDataView trainingdata, DateTime now, ITransformer trainedModel)
        {
            DataViewSchema dataViewSchema = trainingdata.Schema;
            var filename = backupFileLocation + "model" + modelName + now.ToFileTimeUtc();
            using (var fs = File.Create(filename))
            {
                mlContext.Model.Save(trainedModel, dataViewSchema, fs);
            }
            return filename;
        }

        /// <summary>
        /// Displays the training metrics of the trained model.
        /// </summary>
        /// <param name="metrics">The binary classification metrics.</param>
        /// <param name="set">The training set that was used for training</param>
        /// <param name="trainer">The trainer set that was used for training</param>
        private static void DisplayTrainingMetricsMulti(MulticlassClassificationMetrics metrics, string set)
        {
            Console.WriteLine("");
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for classification model      ");
            Console.WriteLine($"*       On training set {set}      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       F1Score:  {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"************************************************************");
            Console.WriteLine("");
            Console.WriteLine("");
        }
    }
}
