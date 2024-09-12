// use burn::tensor::Tensor;
use ndarray::Array2;
use std::collections::HashMap;

#[derive(Debug)]
struct MultinomialNB {
    class_log_prior_: Vec<f64>,
    feature_log_prob_: Vec<Vec<f64>>,
}

impl MultinomialNB {
    fn new() -> Self {
        MultinomialNB {
            class_log_prior_: vec![],
            feature_log_prob_: vec![],
        }
    }

    fn fit(&mut self, x_train: Array2<u64>, y_train: Vec<u64>) {
        let n_samples = x_train.nrows();
        let n_features = x_train.ncols();
        let classes: Vec<u64> = y_train.iter().cloned().collect();

        // Count class frequencies
        let class_count = self.class_count(&y_train);

        // Calculate class prior (log probabilities of each class)
        self.class_log_prior_ = classes
            .iter()
            .map(|&class| (class_count[&class] as f64).ln() - (n_samples as f64).ln())
            .collect();

        // Calculate feature counts by class
        let mut feature_count = vec![vec![0u64; n_features]; classes.len()];

        for (i, class) in y_train.iter().enumerate() {
            for j in 0..n_features {
                feature_count[*class as usize][j] += x_train[[i, j]];
            }
        }

        // Compute smoothed feature log probabilities
        self.feature_log_prob_ = feature_count
            .iter()
            .map(|counts| {
                let total_count: u64 = counts.iter().sum();
                counts
                    .iter()
                    .map(|&count| ((count + 1) as f64).ln() - ((total_count + n_features as u64) as f64).ln())
                    .collect()
            })
            .collect();
    }

    fn predict(&self, x_test: Array2<u64>) -> Vec<u64> {
        let mut predictions = Vec::new();
        let n_features = x_test.ncols();

        for i in 0..x_test.nrows() {
            let mut log_probs: Vec<f64> = Vec::new();

            for (class_index, class_log_prior) in self.class_log_prior_.iter().enumerate() {
                let mut log_prob = *class_log_prior;
                for j in 0..n_features {
                    log_prob += self.feature_log_prob_[class_index][j] * (x_test[[i, j]] as f64);
                }
                log_probs.push(log_prob);
            }

            let predicted_class = log_probs
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index as u64)
                .unwrap();

            predictions.push(predicted_class);
        }

        predictions
    }

    fn class_count(&self, y_train: &Vec<u64>) -> HashMap<u64, usize> {
        let mut class_count = HashMap::new();
        for &label in y_train.iter() {
            *class_count.entry(label).or_insert(0) += 1;
        }
        class_count
    }
}


pub mod tests{
    use super::*;
    pub fn test() {
        // Example training data (term frequencies or counts per document)
        let x_train = Array2::from_shape_vec((4, 3), vec![2, 1, 0, 3, 0, 1, 0, 2, 3, 4, 0, 1]).unwrap();
        let y_train = vec![0, 1, 0, 1];  // Labels
    
        // Example test data
        let x_test = Array2::from_shape_vec((2, 3), vec![1, 0, 1, 3, 2, 1]).unwrap();
    
        // Train Multinomial Naive Bayes
        let mut nb = MultinomialNB::new();
        nb.fit(x_train, y_train);
    
        // Predict classes
        let predictions = nb.predict(x_test);
        assert_eq!(predictions, vec![0, 1]);
    }
}

