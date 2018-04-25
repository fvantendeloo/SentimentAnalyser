package be.kdg;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;


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

        SparkSession spark = SparkSession.builder().getOrCreate();

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

        JavaRDD<Row> words_iterable = emptyStringsRemoved.map(new Function<String, Row>() {
            @Override
            public Row call(String s) throws Exception {
                if (!s.isEmpty() || s.contains(";")) {
                    String[] arr = s.split(";");
                    Row row = RowFactory.create(Double.parseDouble(arr[0]), arr[1].isEmpty() ? "I love Icecream" : arr[1]);
                    return row;
                }
                return RowFactory.create(1.0, "love icecream so much");
            }
        });

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> sentenceData = spark.createDataFrame(words_iterable, schema);

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        int numFeatures = 100;

        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        Dataset<Row> convertVecDF = MLUtils.convertVectorColumnsFromML(rescaledData);
        rescaledData.select("label", "features").show();

        rescaledData.select("label", "features").show();

        // convert ml.linalg.Vector to mllib.linalg.Vector !
        JavaRDD<Row> rescaledRDD = convertVecDF.select("label", "features").toJavaRDD();

        JavaRDD<LabeledPoint> labeledPoints = rescaledRDD.map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {
                return new LabeledPoint(row.getDouble(0), row.getAs(1));
            }
        });

        JavaRDD<LabeledPoint>[] tmp = labeledPoints.randomSplit(new double[]{0.8, 0.2});
        JavaRDD<LabeledPoint> training = tmp[0];
        JavaRDD<LabeledPoint> test = tmp[1];

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel =
                test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double accuracy = predictionAndLabel.filter(pl -> pl._1.equals(pl._2)).count() / (double) test.count();
        System.out.println("Accuracy = " + accuracy);

        model.save(sc.sc(), outputModel);

        NaiveBayesModel myModel = NaiveBayesModel.load(sc.sc(), outputModel);
    }
}
