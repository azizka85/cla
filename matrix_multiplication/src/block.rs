use crate::Number;

pub fn row_blocked_gaxpy<T: Number>(a: &Vec<Vec<T>>, b: &Vec<usize>, x: &Vec<T>, y: &mut Vec<T>) {
    let mut l = 0;

    for k in 0..b.len() {
        for i in l..l+b[k] {
            for j in 0..a[i].len() {
                y[i] += a[i][j] * x[j];
            }
        }

        l += b[k];
    }
}

pub fn column_blocked_gaxpy<T: Number>(a: &Vec<Vec<T>>, b: &Vec<usize>, x: &Vec<T>, y: &mut Vec<T>) {
    let mut l = 0;

    for k in 0..b.len() {
        for i in 0..a.len() {
            for j in l..l+b[k] {
                y[i] += a[i][j] * x[j];
            }            
        }        

        l += b[k];
    }
}

pub fn block_update<T: Number>(c: &mut Vec<Vec<T>>, a: &Vec<Vec<T>>, b: &Vec<Vec<T>>, n: usize, l: usize) {
    for p in 0..n {
        for i in p*l..(p+1)*l {
            for q in 0..n {
                for j in q*l..(q+1)*l {
                    for r in 0..n {
                        for k in r*l..(r+1)*l {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }
}
