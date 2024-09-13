use burn::{backend::Wgpu, tensor::Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct MultinomialNB {
    class_log_prior_: Vec<f64>,
    feature_log_prob_: Vec<Vec<f64>>,
    pub num_classes_: usize,
    pub num_features_: usize,
    pub num_samples_: usize,
}

impl MultinomialNB {
    pub fn new() -> Self {
        MultinomialNB {
            class_log_prior_: vec![],
            feature_log_prob_: vec![],
            num_classes_: 0,
            num_features_: 0,
            num_samples_: 0,
        }
    }

    pub fn fit(&mut self, x_train: &Tensor<Wgpu, 2>, y_train: &Vec<u64>) {
        let shape = x_train.shape();
        self.num_samples_ = shape.dims[0];
        self.num_features_ = shape.dims[1];
        let classes: Vec<u64> = y_train.iter().cloned().collect();
        self.num_classes_ = classes.iter().cloned().collect::<Vec<_>>().len();

        // Get the data from the tensor as a Vec<f64>
        let x_train_data = x_train.to_data().convert::<f64>().value;

        // Count class frequencies
        let class_count = self.class_count(&y_train);

        // Calculate class prior (log probabilities of each class)
        self.class_log_prior_ = classes
            .iter()
            .map(|&class| (class_count[&class] as f64).ln() - (self.num_samples_ as f64).ln())
            .collect();

        // Calculate feature counts by class
        let mut feature_count = vec![vec![0u64; self.num_features_]; classes.len()];

        for (i, class) in y_train.iter().enumerate() {
            for j in 0..self.num_features_ {
                feature_count[*class as usize][j] += x_train_data[i * self.num_features_ + j] as u64;
            }
        }

        // Compute smoothed feature log probabilities
        self.feature_log_prob_ = feature_count
            .iter()
            .map(|counts| {
                let total_count: u64 = counts.iter().sum();
                counts
                    .iter()
                    .map(|&count| ((count + 1) as f64).ln() - ((total_count + self.num_features_ as u64) as f64).ln())
                    .collect()
            })
            .collect();
    }

    pub fn predict(&self, x_test: Tensor<Wgpu, 2>) -> Vec<u64> {
        let mut predictions = Vec::new();
        let shape = x_test.shape();
        let n_features = shape.dims[1];

        // Get the data from the tensor as a Vec<f64>
        let x_test_data = x_test.to_data().convert::<f64>().value;

        for i in 0..shape.dims[0] {
            let mut log_probs: Vec<f64> = Vec::new();

            for (class_index, class_log_prior) in self.class_log_prior_.iter().enumerate() {
                let mut log_prob = *class_log_prior;
                for j in 0..n_features {
                    log_prob += self.feature_log_prob_[class_index][j] * x_test_data[i * n_features + j];
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

// Testing
pub mod tests {
    use super::*;
    use burn::tensor::{Shape, Data};

    pub fn test() {
        // Example training data (term frequencies or counts per document)
        let x_train = Tensor::<Wgpu, 2>::from_data(Data::new(vec![
            2.0, 1.0, 0.0,
            3.0, 0.0, 1.0,
            0.0, 2.0, 3.0,
            4.0, 0.0, 1.0,
        ], Shape::from([4, 3])));

        let y_train = vec![0, 1, 0, 1];  // Labels

        // Example test data
        let x_test = Tensor::<Wgpu, 2>::from_data(Data::new(vec![
            1.0, 0.0, 1.0,
            3.0, 2.0, 1.0,
        ], Shape::from([2, 3])));

        // Train Multinomial Naive Bayes
        let mut nb = MultinomialNB::new();
        nb.fit(&x_train, &y_train);

        // Predict classes
        let predictions = nb.predict(x_test);
        assert_eq!(predictions, vec![0, 1]);
    }
}
