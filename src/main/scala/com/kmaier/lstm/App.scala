package com.kmaier.lstm

import java.io.{BufferedWriter, File, FileWriter}

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

object App {

  def main(args : Array[String]) {
    val basePath = new File("src/main/resources/data/")
    val trainingFiles = new File(basePath, "train/train_")
    val testFiles = new File(basePath, "test/test_")

    val inputString = new File(basePath, "international-airline-passengers.csv")
    val csvLines : List[String] = readCSV(inputString.getAbsolutePath).toList
    val batches : List[List[String]] = csvLines.sliding(12,12).toList

    val numExamples = 12
    val splitPos = math.ceil(numExamples*0.8).toInt
    val (train, test) = batches splitAt splitPos

    // write csv's
    writeCSV(train, trainingFiles.getAbsolutePath)
    writeCSV(test, testFiles.getAbsolutePath)

    val miniBatchSize = 1
    val numPossibleLabels = -1 // for regression
    val labelIndex = 0 // label is in the first csv column
    val regression = true

    // Training Data
    val reader = new CSVSequenceRecordReader()
    reader.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath}%d.csv", 0, 9))

    val trainData = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, regression)

    //Normalize the training data
    val normalizer = new NormalizerStandardize()
    normalizer.fit(trainData)              //Collect training data statistics
    trainData.reset()

    //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
    trainData.setPreProcessor(normalizer);

    // Test Data
    val reader1 = new CSVSequenceRecordReader()
    reader1.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath}%d.csv", 0, 1))

    val testData = new SequenceRecordReaderDataSetIterator(reader1, miniBatchSize, numPossibleLabels, labelIndex, regression)

    //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
    testData.setPreProcessor(normalizer); //Note that we are using the exact same normalization process as the training data

    // ----- Configure the network -----
    val conf : MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(123)    //Random number generator seed for improved repeatability. Optional.
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.005))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
      .build()

    val net : MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()

    net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations

    // ----- Train the network, evaluating the test set performance at each epoch -----
    val nEpochs = 40
    for (i <- 0 to nEpochs) {
      net.fit(trainData)
      //Evaluate on the test set:
      val evaluation : RegressionEvaluation = net.evaluateRegression(testData)
      //println(s"Test set evaluation ${evaluation.getPrecision}, ${evaluation.getCurrentPredictionMean}")

      trainData.reset()
      testData.reset()
      println(s"${net.rnnTimeStep(testData.next().get(0).getFeatures)}")
      println(evaluation.stats())
    }
    println("----- Example Complete -----")
  }

  def writeCSV(batches : List[List[String]], pathname : String) = {
    for((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname+ s"${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      batch.zipWithIndex.map{ case (line, i) =>
        val cols = (s"${line},${i}").split(",").map(_.trim)
        bw.write(List(cols(2),cols(1)).mkString(","))
        bw.newLine()
        //Files.write(Paths.get(pathname+(index)+".csv"), cols(1).getBytes())
      }
      bw.close
    }
  }

  def readCSV(fileName: String) = {
    val bufferedSource = io.Source.fromFile(fileName)
    for {
      line <- bufferedSource.getLines.drop(1)
      cols = line.split(",").map(_.trim)
      if(cols.length > 1)
    } yield line
  }

}
