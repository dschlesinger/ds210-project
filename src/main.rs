mod utils;

use ndarray::{Array1, Array2, Axis, s};
use utils::knn::{KNN, Prediction};
use utils::dataloader::{Iris};

use std::env;

fn main() {

    let args: Vec<String> = env::args().collect();

    let mut n_neighbors: i32 = 32;

    if args.len() > 1 {
        n_neighbors = args[1].parse().unwrap_or(32);
    }

    let data: Iris = Iris::new("data/Iris.csv").expect("Failed to load csv");

    let model = KNN::new(data.x_train.clone(), data.y_train.clone(), n_neighbors);

    let mut total: i32 = 0;

    for (idx, r) in data.x_test.axis_iter(Axis(0)).enumerate() {
        if model.predict(r.clone()) == data.y_test[idx] {
            total += 1;
        }
    }

    println!("KNN is {}% accurate", 100.0 * (total as f32) / data.y_test.len() as f32);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction() {
        /// Checks that prediction function works

        let p = Prediction::new(vec!["Iris-setosa".to_string(), "Iris-setosa".to_string(), "Iris-versicolor".to_string(), "Iris-virginica".to_string()]);

        assert_eq!(p.sample(), "Iris-setosa".to_string());
    }

    #[test]
    fn test_knn() {
        /// Test knns logic works properly

        let mut d: Array2<f32> = Array2::from_shape_vec((5, 5), vec![0.0; 25]).unwrap();

        // Set first two rows to 1
        d.slice_mut(s![0..3, ..]).fill(1.0);

        let l: Vec<String> = vec!["Iris-virginica".to_string(), "Iris-virginica".to_string(), "Iris-virginica".to_string(), "Iris-versicolor".to_string(), "Iris-setosa".to_string()];

        let k = KNN::new(d, l, 5);

        let mut q = Array1::from(vec![1.0; 5]);

        assert_eq!(k.predict(q.view()), "Iris-virginica".to_string());

    }
}
