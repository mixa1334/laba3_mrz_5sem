package com.misha.mrz3.util;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;

public class DataReader {
    private static final Logger logger = LogManager.getLogger();
    private static final String DELIMITER = ", ";

    public static Optional<double[][]> readInputSequences(String path) {
        Optional<double[][]> result = Optional.empty();
        try {
            List<String> strings = Files.readAllLines(Paths.get(path));
            double[][] res = new double[strings.size()][];
            for (int i = 0; i < res.length; i++) {
                String[] values = strings.get(i).split(DELIMITER);
                double[] vector = new double[values.length];
                for (int j = 0; j < values.length; j++) {
                    vector[j] = Double.parseDouble(values[j]);
                }
                res[i] = vector;
            }
            result = Optional.of(res);
        } catch (IOException e) {
            logger.log(Level.ERROR, e.getMessage());
        }

        return result;
    }
}
