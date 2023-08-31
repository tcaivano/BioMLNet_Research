﻿using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;

namespace BioMLNet_Research.Data
{
    static public class DataSetFormatter
    {
        public static void FormatDataSet(string fileLocation)
        {
            var files = Directory.GetFiles(fileLocation, "*.*");

            foreach (var file in files)
            {
                var fileName = Path.GetFileName(file);
                var underscoreIndex = fileName.IndexOf("_", StringComparison.Ordinal);

                if (underscoreIndex <= 0) continue;

                var prefix = fileName.Substring(0, underscoreIndex);

                var newFolderPath = Path.Combine(fileLocation, "subject" + prefix);

                // Create the prefix directory if it doesn't exist
                if (!Directory.Exists(newFolderPath))
                {
                    Directory.CreateDirectory(newFolderPath);
                }

                var newFilePath = Path.Combine(newFolderPath, fileName);

                // Move the file
                File.Move(file, newFilePath);

                using (var image = Image.Load(newFilePath))
                {
                    var pngFileName = Path.ChangeExtension(newFilePath, ".png");
                    image.Save(pngFileName, new PngEncoder());
                    File.Delete(newFilePath);
                }
            }
        }

        internal static void CopyDataSet(string dataSetLocation, string mergedDataSet)
        {
            if (!Directory.Exists(mergedDataSet))
            {
                Directory.CreateDirectory(mergedDataSet);
            }

            // Copy files from source directory to target directory
            foreach (var file in Directory.GetFiles(dataSetLocation))
            {
                var destFile = Path.Combine(mergedDataSet, Path.GetFileName(file));
                File.Copy(file, destFile, true);
            }

            // Copy subdirectories
            foreach (var directory in Directory.GetDirectories(dataSetLocation))
            {
                var destDir = Path.Combine(mergedDataSet, Path.GetFileName(directory));
                CopyDataSet(directory, destDir);
            }
        }

        internal static void CreateAuthDataSet(string dataSetLocation, string authDataSet, int userIndex)
        {
            var subjectDataSet = authDataSet + "\\subject" + userIndex + "\\";
            var otherDataSet = authDataSet + "\\other\\";
            if (!Directory.Exists(authDataSet))
            {
                Directory.CreateDirectory(authDataSet);
            }

            if (!Directory.Exists(subjectDataSet))
            {
                Directory.CreateDirectory(subjectDataSet);
            }

            if (!Directory.Exists(otherDataSet))
            {
                Directory.CreateDirectory(otherDataSet);
            }

            // Copy files from source directory to target directory
            foreach (var file in Directory.GetFiles(dataSetLocation, "*.*", SearchOption.AllDirectories))
            {
                string destFile;
                if (Path.GetFileName(file).StartsWith(userIndex + "_"))
                {
                    destFile = Path.Combine(subjectDataSet, Path.GetFileName(file));
                }
                else
                {
                    destFile = Path.Combine(otherDataSet, Path.GetFileName(file));
                }

                File.Copy(file, destFile);
            }
        }
    }
}
