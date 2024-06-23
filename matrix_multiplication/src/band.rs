use crate::Number;

pub fn band_gaxpy<T: Number>(a: &Vec<Vec<T>>, p: usize, q: usize, x: &Vec<T>, y: &mut Vec<T>) {
    if !a.is_empty() {
        let n = a[0].len();

        for i in 0..n {        
            let j1 = 0.max(i as i32 - q as i32) as usize;
            let j2 = (n - 1).min(i + p);
            
            let k = 0.max(q as i32 - i as i32) as usize;        
    
            for j in j1..=j2 {
                y[j] += a[k+j-j1][i] * x[i];
            }
        }
    }    
}
