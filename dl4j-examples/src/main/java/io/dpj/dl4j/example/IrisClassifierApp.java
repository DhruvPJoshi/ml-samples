package io.dpj.dl4j.example;

import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Classification of basic 'Iris flowers' dataset
 * 
 * https://www.baeldung.com/deeplearning4j
 * 
 */
public class IrisClassifierApp 
{

    // each dataset row contains 4 fields
    private static final int FEATURE_COUNT = 4;
    // each dataset row can be classified in 3 categories
    private static final int CLASSES_COUNT = 3;

    public static void main( String[] args )
        throws IOException, InterruptedException {

        DataSet allData;

        // load iris cvs dataset and store it in an in-memory dataset
        try(RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(
                new ClassPathResource("iris.txt").getFile()));

            // dataset contains total of 150 rows
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                recordReader, 150, FEATURE_COUNT, CLASSES_COUNT);

            allData = iterator.next();
        }

        // constant seed amount ensures the reordered dataset on every run
        allData.shuffle(42);

        DataNormalization normalizer = new NormalizerStandardize();
        // gather statistics of dataset
        normalizer.fit(allData);
        // make dataset uniform
        normalizer.transform(allData);

        // use 75% data to train the model, rest 25% for testing the model
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration
            .Builder()
                .iterations(1000)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer
                    .Builder().nIn(FEATURE_COUNT).nOut(3).build())
                .layer(1, new DenseLayer
                    .Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer
                    .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3)
                        .nOut(CLASSES_COUNT)
                        .build())
                .backprop(true).pretrain(false)
                .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(trainingData);

        INDArray output =  model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());

        System.exit(0);
    }
}
