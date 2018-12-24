package com.kmaier.lstm

import java.io.{BufferedWriter, File, FileWriter}

import com.kmaier.lstm.plotting.PlotUtil
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
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
    val csvLines1 = csvLines zip csvLines.tail
    val batches : List[List[(String,String)]] = csvLines1.sliding(1,1).toList // .sliding(12,12) results in batches of 12 months

    val numExamples = batches.length
    val splitPos = math.ceil(numExamples*0.8).toInt
    val (train, test) = batches splitAt splitPos

    // write csv's
    writeCSV(train, trainingFiles.getAbsolutePath)
    writeCSV(test, testFiles.getAbsolutePath)

    val miniBatchSize = 1
    val numPossibleLabels = -1 // for regression
    val labelIndex = 1 // label is in the first csv column
    val regression = true

    // Training Data
    // each line in the csv data represents one time step, with the first row as earliest time series observation
    val reader = new CSVSequenceRecordReader()
    reader.initialize(new NumberedFileInputSplit(s"${trainingFiles.getAbsolutePath}%d.csv", 0, 114))

    val trainData = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, regression)

    //Normalize the training data
    val normalizer = new NormalizerStandardize()
    normalizer.fit(trainData)              //Collect training data statistics
    trainData.reset()

    //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
    trainData.setPreProcessor(normalizer);

    // Test Data
    val reader1 = new CSVSequenceRecordReader()
    reader1.initialize(new NumberedFileInputSplit(s"${testFiles.getAbsolutePath}%d.csv", 0, 26))

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
      .layer(0, new LSTM.Builder()
        .activation(Activation.TANH)
        .nIn(1)
        .nOut(10).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(10)
        .nOut(1).build())
      .build()

    val net : MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()

    //net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations

    //Initialize the user interface backend
    val uiServer = UIServer.getInstance();
    val statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage)
    net.setListeners(new StatsListener(statsStorage));

    net.addListeners(new ScoreIterationListener(100));

    // ----- Train the network, evaluating the test set performance at each epoch -----
    val nEpochs = 20000
    for (i <- 0 to nEpochs) {
      net.fit(trainData)
      //Evaluate on the test set:
      val evaluation : RegressionEvaluation = net.evaluateRegression(testData)
      println(s"======== EPOCH ${i} ========")
      println(evaluation.stats())
      trainData.reset()
      testData.reset()

      var predicts : Array[Double] = Array()
      var actuals : Array[Double] = Array()

      while(testData.hasNext) {
        //net.rnnClearPreviousState()
        val nextTestPoint = testData.next
        val nextTestPointFeatures = nextTestPoint.getFeatures
        val predictionNextTestPoint : INDArray = net.rnnTimeStep(nextTestPointFeatures)//net.rnnTimeStep(nextTestPointFeatures) // net.output(nextTestPointFeatures)

        val nextTestPointLabels = nextTestPoint.getLabels
        normalizer.revert(nextTestPoint) // revert the normalization of this test point
        println(s"Test point no.: ${nextTestPointFeatures} \n" +
        s"Prediction is: ${predictionNextTestPoint} \n" +
        s"Actual value is: ${nextTestPointLabels} \n")
        predicts = predicts :+ predictionNextTestPoint.getDouble(0L)
        actuals = actuals :+ nextTestPointLabels.getDouble(0L)
      }
      if(i % 1000 == 0)
        PlotUtil.plot(predicts, actuals, s"Test Run", i)

      testData.reset()
      //net.rnnClearPreviousState()
    }

    println("----- Example Complete -----")
  }

  /**
    * Writes .csv files (train/test) according to the batches inputs
    * @param batches
    * @param pathname
    */
  def writeCSV(batches : List[List[(String,String)]], pathname : String) = {
    for((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname+ s"${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      batch.map{ case (line, i) =>
        val cols = (s"${line},${i}").split(",").map(_.trim)
        bw.write(List(cols(0),cols(1)).mkString(","))
        bw.newLine()
        //Files.write(Paths.get(pathname+(index)+".csv"), cols(1).getBytes())
      }
      bw.close
    }
  }

  /**
    *
    * @param fileName
    * @return Iterator over all lines found in the .csv with an absolute index
    */
  def readCSV(fileName: String) = {
    val bufferedSource = scala.io.Source.fromFile(fileName)
    for {
      line <- bufferedSource.getLines.drop(1)
      cols = line.split(",").map(_.trim)
      if(cols.length > 1) // neglect lines that don't hold data
    } yield cols(1)
  }

}
