package com.misha.mrz3;

import com.misha.mrz3.network.JordanNeuralNetwork;
import com.misha.mrz3.util.DataReader;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;

public class Main {
    private static final Logger logger = LogManager.getLogger();

    public static void main(String[] args) {
        JordanNeuralNetwork network = new JordanNeuralNetwork(0.0001, 0.0000005, 1_000_000, 4, 2);

        double[][] inputs = DataReader.readInputSequences("src/main/resources/input.txt").get();

        for (double[] input : inputs) {
            logger.log(Level.INFO, "input -> " + Arrays.toString(input) + "\n");
            network.initialize(input);
            network.learn();
            double[] result = network.predictTheSequence(5);
            logger.log(Level.INFO, "result -> " + Arrays.toString(result) + "\n");
            logger.log(Level.INFO, "----------------------------------------");
        }
    }
}
