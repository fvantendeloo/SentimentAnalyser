package be.kdg;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Function1;

import java.util.Arrays;

/**
 * @author Floris Van Tendeloo
 */
public class NaiveBayesModeller {

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

        //cleaned.collect().forEach(System.out::println);

        JavaRDD<Row> words_iterable = cleaned.map(new Function<String, Row>() {
            @Override
            public Row call(String s) throws Exception {
                System.out.println(s);
                String[] arr = s.split(";");
                Row row = RowFactory.create(Double.parseDouble(arr[0]), arr[1]);
                System.out.println(row);
                return row;
            }
        });

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> sentenceData = spark.createDataFrame(words_iterable, schema);

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        int numFeatures = 1000;

        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        rescaledData.select("label", "features").show();

        //cleaned.collect().forEach(l -> System.out.println(l));
    }
}
