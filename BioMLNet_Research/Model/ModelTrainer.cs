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
        public static string TrainModel(string modelLocation, float testFraction, float learningRate, int batchSize, int? epochs, ImageClassificationTrainer.Architecture arch, string saveLocation)
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


            IDataView trainSet;
            IDataView validationSet;
            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: testFraction);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);
            trainSet = trainSplit.TrainSet;
            validationSet = validationTestSplit.TrainSet;

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = arch,
                MetricsCallback = (metrics) => Console.Write("\r" + metrics),
                TestOnTrainSet = true,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
                LearningRate = learningRate,
                BatchSize = batchSize
            };

            if (epochs != null)
            {
                classifierOptions.Epoch = (int)epochs;
                classifierOptions.EarlyStoppingCriteria = null;
            }

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
            IEnumerable<ImageData> images = LoadImagesFromDirectory(dataSet);

            int correct = 0, incorrect = 0, i = 0;
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



        internal static void EvaluateModelAuthentication(string modelLocation, string dataSet)
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
            IEnumerable<ImageData> images = LoadImagesFromDirectory(dataSet);

            int tp = 0, tn = 0, fp = 0, fn = 0, i = 0;
            int total = images.Count();
            foreach (var image in images)
            {
                var input = new ModelInput()
                {
                    Label = image.Label,
                    LabelAsKey = image.Label == "other" ? 0 : UInt32.Parse(image.Label.Remove(0, 7)),
                    Image = System.IO.File.ReadAllBytes(image.ImagePath),
                    ImagePath = image.ImagePath
                };
                var p = predictionEngine.Predict(input);
                if (image.Label == p.PredictedLabel && image.Label != "other")
                {
                    tp++;
                }
                else if (image.Label == p.PredictedLabel && image.Label == "other")
                {
                    tn++;
                }
                else if (image.Label != p.PredictedLabel && image.Label != "other")
                {
                    fn++;
                }
                else
                {
                    fp++;
                }
                i++;
                Console.Write($"\rTotal: {total}, Index {i}, True Positive: {tp}, False Negative: {fn}, True Negative: {tn}, False Positive {fp}");
            }

            Console.WriteLine($"\rTotal: {total}, Index {i}, True Positive: {tp}, False Negative: {fn}, True Negative: {tn}, False Positive {fp}");
            float pr = (float)tp / (float)((float)tp + (float)fp);
            float r = (float)tp / ((float)tp + (float)fn);
            float f1 = (2 * pr * r) / (pr + r);
            Console.WriteLine($"F1 Score: {f1}");
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
    }
}
