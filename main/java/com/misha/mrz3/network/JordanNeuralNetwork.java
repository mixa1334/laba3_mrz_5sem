package com.misha.mrz3.network;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class JordanNeuralNetwork {
    private static final Logger logger = LogManager.getLogger();

    private final double a, E;

    private final int i, m, p;

    private double[] X;

    private double C;

    private double[][] W1, W2;

    public JordanNeuralNetwork(double a, double E, int i, int m, int p) {
        this.a = a;
        this.E = E;
        this.i = i;
        this.m = m;
        this.p = p;
    }

    public void initialize(double[] X) {
        this.X = X;
        W1 = new double[m + 2][p];
        W2 = new double[p + 1][1];

        for (int i = 0; i < W1.length; i++) {
            for (int j = 0; j < W1[i].length; j++) {
                W1[i][j] = Math.random() * 2 - 1;
            }
        }

        for (int i = 0; i < W2.length; i++) {
            for (int j = 0; j < W2[i].length; j++) {
                W2[i][j] = Math.random() * 2 - 1;
            }
        }

//        String W1String = matrixToString(W1);
//        String W2String = matrixToString(W2);
//        logger.log(Level.DEBUG, "W1 -> \n" + W1String + "\n W2 -> \n" + W2String + "\n");
    }

    public void learn() {
        int iteration = 0;
        double[] vector = new double[p];

        double error;
        do {
            error = 0;
            C = 0;

            for (int i = 0; i < 2; i++) {
                for (int offset = 0; offset < X.length - m; offset++) {

                    for (int j = 0; j < p; j++) {
                        vector[j] = 0;
                        vector[j] += C * W1[0][j];
                        vector[j] += 1 * W1[W1.length - 1][j];
                        for (int k = 0; k < m; k++) {
                            vector[j] += X[offset + k] * W1[k + 1][j];
                        }
                        vector[j] = Math.max(vector[j] * 0.01, vector[j]);
                    }

                    C = 0;
                    for (int k = 0; k < p; k++) {
                        C += vector[k] * W2[k][0];
                    }
                    C += 1 * W2[W2.length - 1][0];
                    C = Math.max(C * 0.01, C);

                    double linearError = C - X[offset + m];
                    if (i == 0) {
                        adjustWeights(offset, linearError, vector);
                    } else {
                        error += linearError * linearError / 2;
                    }
                }
            }

            iteration++;

//            logger.log(Level.DEBUG, "Error " + error + " context neuron " + C + " iteration " + iteration);
        } while (iteration < i && error > E);
    }

    public double[] predictTheSequence(int number) {
        double[] vector = new double[p];
        double[] result = new double[X.length + number];
        int n = 0;
        while (n < X.length) {
            result[n] = X[n];
            n++;
        }

        for (int i = 0; i < number; i++) {

            for (int j = 0; j < p; j++) {
                vector[j] = 0;
                vector[j] += C * W1[0][j];
                vector[j] += 1 * W1[W1.length - 1][j];
                for (int k = 0; k < m; k++) {
                    vector[j] += result[n - m + k] * W1[k + 1][j];
                }
                vector[j] = Math.max(vector[j] * 0.01, vector[j]);
            }

            C = 0;
            for (int k = 0; k < p; k++) {
                C += vector[k] * W2[k][0];
            }
            C += 1 * W2[W2.length - 1][0];
            C = Math.max(C * 0.01, C);

            result[n++] = C;
        }


        String W1String = matrixToString(W1);
        String W2String = matrixToString(W2);
        logger.log(Level.INFO, "\nW1 -> \n" + W1String + "\n W2 -> \n" + W2String);

        return result;
    }

    private void adjustWeights(int offset, double error, double[] vector) {
        for (int i = 0; i < m + 2; i++) {
            for (int j = 0; j < p; j++) {
                double value;
                if (i == m) {
                    value = C;
                } else if (i > m) {
                    value = 1;
                } else {
                    value = X[offset + i];
                }

                W1[i][j] = W1[i][j] - a * value * error * W2[j][0];
            }
        }

        for (int i = 0; i < p + 1; i++) {
            double value = i != p ? vector[i] : 1;
            W2[i][0] = W2[i][0] - a * value * error;
        }
    }

    private String matrixToString(double[][] matrix) {
        StringBuilder builder = new StringBuilder();
        for (double[] vector : matrix) {
            for (double value : vector) {
                builder.append(value).append("   ");
            }
            builder.append("\n");
        }
        return builder.toString();
    }
}