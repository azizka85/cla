pub mod band;
pub mod block;

pub trait Number: std::ops::Mul<Self, Output = Self> + std::ops::AddAssign<Self> 
					+ Copy + Default {}

impl Number for i64 {}
impl Number for f64 {}

pub fn dot_product<T: Number>(x: &Vec<T>, y: &Vec<T>) -> T {
	let mut r = Default::default();

	for i in 0..x.len() {
		r += x[i] * y[i];
	}

	r
}
  
pub fn saxpy<T: Number>(a: T, x: &Vec<T>, y: &mut Vec<T>) {
	for i in 0..y.len() {
		y[i] += a * x[i];
	}
}

pub fn transpose<T: Number>(a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
	if a.is_empty() {
		return Vec::new();
	}

	let m = a.len();
	let n = a[0].len();

	let mut t = vec![vec![Default::default(); m]; n];

	for i in 0..m {
		for j in 0..n {
			t[j][i] = a[i][j];
		}
	}

	t
}

pub fn row_oriented_gaxpy<T: Number>(a: &Vec<Vec<T>>, x: &Vec<T>, y: &mut Vec<T>) {
	for i in 0..y.len() {
		for j in 0..x.len() {
			y[i] += a[i][j] * x[j];
		}
	}
}

pub fn column_oriented_gaxpy<T: Number>(a: &Vec<Vec<T>>, x: &Vec<T>, y: &mut Vec<T>) {
	for j in 0..x.len() {
		for i in 0..y.len() {
			y[i] += a[i][j] * x[j];
		}
	}
}

pub fn row_oriented_outer_product_update<T: Number>(a: &mut Vec<Vec<T>>, x: &Vec<T>, y: &Vec<T>) {
	for i in 0..x.len() {
		for j in 0..y.len() {
			a[i][j] += x[i] * y[j];
		}
	}
}

pub fn column_oriented_outer_product_update<T: Number>(a: &mut Vec<Vec<T>>, x: &Vec<T>, y: &Vec<T>) {
	for j in 0..y.len() {
		for i in 0..x.len() {
			a[i][j] += x[i] * y[j];
		}
	}
}
