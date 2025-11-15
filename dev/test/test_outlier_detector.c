/*
 * Outlier Detector Testing Suite
 * ===============================
 *
 * This test validates the OutlierDetector, a critical component for
 * detecting anomalies during neural network training.
 *
 * WHAT IS THE OUTLIER DETECTOR?
 * The OutlierDetector monitors a stream of values (typically training loss)
 * and computes a running z-score to identify outliers. A z-score measures
 * how many standard deviations a value is from the mean.
 *
 * WHY IS THIS IMPORTANT?
 * During training, sudden spikes in loss can indicate:
 * - Numerical instability (NaNs, infinities)
 * - Bad data batches
 * - Learning rate problems
 * - Gradient explosion
 *
 * Detecting these outliers early allows the training loop to:
 * - Skip bad batches
 * - Reduce learning rate
 * - Restart from a checkpoint
 * - Alert the user
 *
 * HOW IT WORKS:
 * The detector maintains a sliding window of recent values and computes:
 * - Running mean: average of values in the window
 * - Running standard deviation: spread of values in the window
 * - Z-score: (current_value - mean) / std_dev
 *
 * A z-score > 3 or < -3 indicates a likely outlier (beyond 3 standard deviations).
 *
 * TEST STRATEGY:
 * 1. Feed normal random values and verify z-scores are reasonable
 * 2. Feed an outlier (10.0) and verify it produces a high z-score
 * 3. Verify the detector returns NaN until the window fills up
 *
 * COMPILE AND RUN:
 * From dev/test directory:
 *   gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
 */

#include <stdlib.h>
#include "../../llmc/outlier_detector.h"  // OutlierDetector implementation

/*
 * Main Test Program
 * =================
 *
 * Tests the OutlierDetector by:
 * 1. Feeding it normal random values
 * 2. Verifying z-scores behave correctly during warm-up
 * 3. Verifying z-scores are reasonable for normal values
 * 4. Feeding it an outlier and verifying detection
 */
int main(void) {
    // ========================================================================
    // SETUP: Initialize the detector
    // ========================================================================
    OutlierDetector detector;
    init_detector(&detector);

    // Seed RNG for reproducible test
    srand(1337);

    // ========================================================================
    // PHASE 1: Normal values test (warm-up + steady state)
    // ========================================================================
    // Feed the detector 2x the window size of normal random values
    // This tests both the warm-up period and steady-state operation
    for (int i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        // Generate random value in [-1, 1]
        // This simulates normal training loss fluctuations
        double val = (double)rand() / RAND_MAX * 2 - 1;

        // Update detector with this value and get z-score
        double zscore = update_detector(&detector, val);

        // Print for debugging/inspection
        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // VALIDATION 1: Check warm-up period behavior
        // During the first OUTLIER_DETECTOR_WINDOW_SIZE steps, the window
        // isn't full yet, so the detector should return NaN (not enough data
        // to compute meaningful statistics)
        if (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            if (!isnan(zscore)) {
                printf("Error: Expected nan during warm-up, got %.4f\n", zscore);
                return EXIT_FAILURE;
            }
        } else {
            // VALIDATION 2: Check steady-state behavior
            // After warm-up, z-scores for normal random values should be
            // within reasonable bounds. The 3-sigma rule says 99.7% of values
            // in a normal distribution fall within [-3, 3] z-scores.
            // Our random values are uniform (not normal), but they should
            // still have reasonable z-scores.
            if (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                return EXIT_FAILURE;
            }
        }
    }

    // ========================================================================
    // PHASE 2: Outlier detection test
    // ========================================================================
    // Simulate a sudden loss spike (e.g., from gradient explosion or bad batch)
    // The value 10.0 is much larger than our normal range of [-1, 1]
    double outlier = 10.0;  // <--- Simulated loss spike
    double zscore = update_detector(&detector, outlier);

    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // VALIDATION 3: Verify outlier is detected
    // The z-score for an outlier should be large (> 5 in this case)
    // This confirms the detector successfully identifies anomalies
    if (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        return EXIT_FAILURE;
    }

    // ========================================================================
    // All tests passed!
    // ========================================================================
    printf("OK\n");
    return EXIT_SUCCESS;
}
