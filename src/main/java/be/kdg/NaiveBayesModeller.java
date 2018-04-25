package be.kdg;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;
import java.util.Arrays;



/**
 * @author Floris Van Tendeloo
 */
public class NaiveBayesModeller {

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        if (args.length < 3 || args.length > 4) {
            System.err.println("Usage: " +
                    "1 -> training file location " +
                    "2 -> output file location " +
                    "3 -> batch duration " +
                    "4 -> (forceLocal)");
            System.exit(1);
        }

        SparkConf conf;
        JavaSparkContext sc;
        String trainingData = args[0];
        String outputModel = args[1];
        Stemmer stemmer = new Stemmer();

        conf = new SparkConf().setAppName("NaiveBayesModeller");

        if (args.length == 4 && args[3].toLowerCase().equals("forcelocal")) {
            conf.setMaster("local[2]");
        }

        sc = new JavaSparkContext(conf);

        JavaRDD<String> raw = sc.textFile(trainingData, 3);
        JavaRDD<String> cleaned = raw.map(l -> {
            String[] arr = l.split(",");
            String punctuationRemoved = arr[3].toLowerCase().replaceAll("\\p{Punct}", "");

            String stopwordsRemoved = StopwordsRemover.removeStopwords(punctuationRemoved);
            String replaceWhitespaces = stopwordsRemoved.replaceAll("\\s{2,}", " ");
            String result = arr[1] + ";";
            for (String str : replaceWhitespaces.split(" ")) {
                if (!str.startsWith("http")) {
                    result += stemmer.stem(str) + " ";
                }
            }
            return result;
        });

        JavaRDD<String> emptyStringsRemoved = cleaned.filter(s -> !s.isEmpty() && !(s.length() <= 2));

        HashingTF tf = new HashingTF();
        JavaRDD<LabeledPoint> points = emptyStringsRemoved.map(new Function<String, LabeledPoint>() {
            @Override
            public LabeledPoint call(String s) throws Exception {
                String[] arr = s.split(";");
                return new LabeledPoint(Double.parseDouble(arr[0]), tf.transform(Arrays.asList(arr[1].split(" "))));
            }
        });

        JavaRDD<LabeledPoint>[] tmp = points.randomSplit(new double[]{0.8, 0.2});
        JavaRDD<LabeledPoint> training = tmp[0];
        JavaRDD<LabeledPoint> test = tmp[1];

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel =
                test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double accuracy = predictionAndLabel.filter(pl -> pl._1.equals(pl._2)).count() / (double) test.count();
        System.out.println("Accuracy = " + accuracy);

        model.save(sc.sc(), outputModel);
    }
}
