import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;

// Semisupervised Tomatoes:
// EM some Naive Bayes and Markov Models to do sentiment analysis.
// Based on solution code for Assignment 3.
//
// Input from train.tsv.zip at 
// https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
//
// itself gathered from Rotten Tomatoes.
//
// Format is PhraseID[unused]   SentenceID  Sentence[tokenized]
//
// Just a few sentiment labels this time - this is semisupervised.
//
// We'll only use the first line for each SentenceID, since the others are
// micro-analyzed phrases that would just mess up our counts.
//
// After training, we'll identify the top words for each cluster by
// Pr(cluster | word) - the words that are much more likely in the cluster
// than in the general population - and categorize the new utterances.

public class SemisupervisedTomatoes {

  public static final int CLASSES = 2;
  // Assume sentence numbering starts with this number in the file
  public static final int FIRST_SENTENCE_NUM = 1;

  // Probability of either a unigram or bigram that hasn't been seen
  // Gotta make this real generous if we're not using logs
  public static final double OUT_OF_VOCAB_PROB = 0.000001;

  // Words to print per class
  public static final int TOP_N = 10;
  // Times (in expectation) that we need to see a word in a cluster
  // before we think it's meaningful enough to print in the summary
  public static final double MIN_TO_PRINT = 15.0;

  public static boolean USE_UNIFORM_PRIOR = false;
  public static boolean SEMISUPERVISED = true;
  public static boolean FIXED_SEED = true;

  public static final int ITERATIONS = 200;

  // We may play with this in the assignment, but it's good to have common
  // ground to talk about
  public static Random rng = (FIXED_SEED ? new Random(2018) : new Random());

  public static NaiveBayesModel nbModel;

  public static class NaiveBayesModel {
    public double[] classCounts;
    public double[] totalWords;
    public ArrayList<HashMap<String, Double>> wordCounts;

    public NaiveBayesModel() {
      classCounts = new double[CLASSES];
      totalWords = new double[CLASSES];
      wordCounts = new ArrayList<HashMap<String, Double>>();
      for (int i = 0; i < CLASSES; i++) {
        wordCounts.add(new HashMap<String, Double>());
      }
    }

    // Update the model given a sentence and its probability of
    // belonging to each class
    void update(String sentence, ArrayList<Double> probs) {

      //strip away class markers if necessary.
      if (sentence.startsWith(":(") || sentence.startsWith(":)")){
        sentence = sentence.substring(3);
      }

      //breakdown given sentence into space delimited tokens
      //that are then converted into lower case.
      String[] sentenceArray = stringSplitter(sentence);

      //update all classes.
      for (int i = 0; i < CLASSES; i++) {
        //for the given class, increase classCounts and
        //totalWord counts appropriately.
        classCounts[i] += probs.get(i);
        totalWords[i] += (probs.get(i) * sentenceArray.length);


        //secure the wordCount hashMap for the given class
        HashMap currentClassWordMap = wordCounts.get(i);

        /*for each word in the sentence, first query to see if the word exists
        in the secured map. If not, add the word, if so, update the word.
        Use the given prob value for either addition or initialization.
        */
        for (String currentWord : sentenceArray) {
          double currentWordCount = 0;
          if (currentClassWordMap.containsKey(currentWord)) {
            currentWordCount = (double) currentClassWordMap.get(currentWord) + probs.get(i);
          } else {
            currentWordCount = probs.get(i);
          }
          currentClassWordMap.put(currentWord, currentWordCount);
        }
      }
    }

    //small class that creates a String array based on the original sentence delimited by spaces
    //and then converted all to lower case.
    private static String[] stringSplitter(String sentence) {
      String[] sentenceArray = sentence.split(" ");
      for (String s : sentenceArray) {
        s.toLowerCase();
      }
      return sentenceArray;
    }

    // Classify a new sentence using the data and a Naive Bayes model.
    // Assume every token in the sentence is space-delimited, as the input
    // was.  Return a list of class probabilities.
    public ArrayList<Double> classify(String sentence) {
      //create new output ArrayList
      ArrayList<Double> output = new ArrayList<>();

      //for use in keeping track of what sentiment is currently being checked.
      int classNum = 0;

      //find total value of all classes for use in individual class probability calc.
      double totalClassValue = 0;
      for (Double classValue : classCounts) {
        totalClassValue += classValue;
      }

      //strip away class marker if necessary.
      if (sentence.startsWith(":(") || sentence.startsWith(":)")){
        sentence = sentence.substring(3);
      }

      //break down the sentence into a lower case array
      String[] sentenceArray = stringSplitter(sentence);

      //iterate over every class's word bank.
      for (HashMap h : wordCounts) {
        //calculate the native proportion of the class.
        double classValue = classCounts[classNum] / totalClassValue;
        double sentenceClassValue = 1;

        //retrieve and calculate values for each word in sentence.
        for (String s : sentenceArray) {
          //Use default value if not found in the hashMap.
          double wordValue = OUT_OF_VOCAB_PROB;
          //if word found, find value of word given the sentiment.
          if (h.containsKey(s)) {
            //wordValue = (value of word to a given class) / (total value of all words of a class)
            wordValue = (double) h.get(s) / totalWords[classNum];
          }
          //Multiply previous word value and replace 0's with very small fractions
          sentenceClassValue *= wordValue;
          if (sentenceClassValue == 0){
            sentenceClassValue = Double.MIN_NORMAL;
          }
        }

        //multiply in the native value of the class to the total and limit 0's again.
        sentenceClassValue *= classValue;
        if (sentenceClassValue == 0){
          sentenceClassValue = Double.MIN_NORMAL;
        }
        output.add(sentenceClassValue);
        //Check next sentiment;
        classNum++;
      }

      //combine all the sentence's class values to use as normalizing denominator.
      double totalOutput = 0;
      for (double d : output){
        totalOutput+= d;
      }

      //normalize all class values into probabilities that sum to 1.
      for (int i = 0; i < output.size(); i++){
        double normalizedClassProb = output.get(i) / totalOutput;
        output.set(i, normalizedClassProb);
      }

      //return the list of normalized probabilities for the sentence.
      return output;
    }

    // printTopWords: Print five words with the highest
    // Pr(thisClass | word) = scale Pr(word | thisClass)Pr(thisClass)
    // but skip those that have appeared (in expectation) less than
    // MIN_TO_PRINT times for this class (to avoid random weird words
    // that only show up once in any sentence)
    void printTopWords(int n) {
      for (int c = 0; c < CLASSES; c++) {
        System.out.println("Cluster " + c + ":");
        ArrayList<WordProb> wordProbs = new ArrayList<WordProb>();
        for (String w : wordCounts.get(c).keySet()) {
          if (wordCounts.get(c).get(w) >= MIN_TO_PRINT) {
            // Treating a word as a one-word sentence lets us use
            // our existing model
            ArrayList<Double> probs = nbModel.classify(w);
            wordProbs.add(new WordProb(w, probs.get(c)));
          }
        }
        Collections.sort(wordProbs);
        for (int i = 0; i < n; i++) {
          if (i >= wordProbs.size()) {
            System.out.println("No more words...");
            break;
          }
          System.out.println(wordProbs.get(i).word);
        }
      }
    }
  }

  public static void main(String[] args) throws FileNotFoundException {
    //Scanner myScanner = new Scanner(System.in);
    Scanner myScanner = new Scanner(new File("trainEMsemisup.txt"));
    ArrayList<String> sentences = getTrainingData(myScanner);
    trainModels(sentences);
    nbModel.printTopWords(TOP_N);
    classifySentences(myScanner);
  }

  public static ArrayList<String> getTrainingData(Scanner sc) {
    int nextFresh = FIRST_SENTENCE_NUM;
    ArrayList<String> sentences = new ArrayList<String>();
    while (sc.hasNextLine()) {
      String line = sc.nextLine();
      if (line.startsWith("---")) {
        return sentences;
      }
      // Data should be filtered now, so just add it
      sentences.add(line);
    }
    return sentences;
  }

  static void trainModels(ArrayList<String> sentences) {
    // We'll start by assigning the sentences to random classes.
    // 1.0 for the random class, 0.0 for everything else
    System.err.println("Initializing models....");
    HashMap<String, ArrayList<Double>> naiveClasses = randomInit(sentences);
    // Initialize the parameters by training as if init were
    // ground truth
    //updateModels(naiveClasses);
    nbModel = new NaiveBayesModel();

    //initialize model with naive classifications.
    for (String currentSentence: sentences){
      if (currentSentence.startsWith(":(") || currentSentence.startsWith(":)")){
        currentSentence = currentSentence.substring(3);
      }
      ArrayList<Double> currentSentenceProbs = naiveClasses.get(currentSentence);
      nbModel.update(currentSentence, currentSentenceProbs);
    }

    //begin EM process.
    for (int i = 0; i < ITERATIONS; i++) {
      System.err.println("EM round " + i);

      //Compute expected probabilities of all sentences using the current model
      //and update naiveClasses with the new values.
      ArrayList<ArrayList<Double>> sentenceClassifications = new ArrayList<>();
      for (String currentSentence: sentences){
        ArrayList<Double> currentSentenceClassifications = nbModel.classify(currentSentence);
        naiveClasses.put(currentSentence, currentSentenceClassifications);
      }
      //with the updated naiveClasses, maximize the model's counts.
      for (String currentSentence: sentences){
        ArrayList<Double> currentSentenceProbs = naiveClasses.get(currentSentence);
        nbModel.update(currentSentence, currentSentenceProbs);
      }
    }
  }


  static HashMap<String, ArrayList<Double>> randomInit(ArrayList<String> sents) {
    HashMap<String, ArrayList<Double>> counts = new HashMap<String, ArrayList<Double>>();
    for (String sent : sents) {
      ArrayList<Double> probs = new ArrayList<Double>();
      if (SEMISUPERVISED && sent.startsWith(":)")) {
        // Class 1 = positive
        probs.add(0.0);
        probs.add(1.0);
        for (int i = 2; i < CLASSES; i++) {
          probs.add(0.0);
        }
        // Shave off emoticon
        sent = sent.substring(3);
      } else if (SEMISUPERVISED && sent.startsWith(":(")) {
        // Class 0 = negative
        probs.add(1.0);
        probs.add(0.0);
        for (int i = 2; i < CLASSES; i++) {
          probs.add(0.0);
        }
        // Shave off emoticon
        sent = sent.substring(3);
      } else {
        double baseline = 1.0 / CLASSES;
        // Slight deviation to break symmetry
        int randomBumpedClass = rng.nextInt(CLASSES);
        double bump = (1.0 / CLASSES * 0.25);
        if (SEMISUPERVISED) {
          // Symmetry breaking not necessary, already got it
          // from labeled examples
          bump = 0.0;
        }
        for (int i = 0; i < CLASSES; i++) {
          if (i == randomBumpedClass) {
            probs.add(baseline + bump);
          } else {
            probs.add(baseline - bump / (CLASSES - 1));
          }
        }
      }
      counts.put(sent, probs);
    }
    return counts;
  }

  public static class WordProb implements Comparable<WordProb> {
    public String word;
    public Double prob;

    public WordProb(String w, Double p) {
      word = w;
      prob = p;
    }

    public int compareTo(WordProb wp) {
      // Reverse order
      if (this.prob > wp.prob) {
        return -1;
      } else if (this.prob < wp.prob) {
        return 1;
      } else {
        return 0;
      }
    }
  }

  public static void classifySentences(Scanner scan) {
    while (scan.hasNextLine()) {
      String line = scan.nextLine();
      System.out.print(line + ":");
      ArrayList<Double> probs = nbModel.classify(line);
      for (int c = 0; c < CLASSES; c++) {
        System.out.print(probs.get(c) + " ");
      }
      System.out.println();
    }
  }

}