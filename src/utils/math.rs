// All math equations only ended up using euclidian distance

use ndarray::{ArrayView1, Array1, Array2};

pub fn euc(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    /// Finds euclidian distance
    /// [vec1, vec2] -> ((vec1 - vec2) ** 2) ** 1/2

    a.iter()
     .zip(b.iter())
     .map(|(&x, &y)| (x - y).powi(2))
     .sum::<f32>()
     .sqrt()
}